#!/usr/bin/env bun
/**
 * MCP stdio-to-HTTP proxy plugin.
 *
 * Reads JSON-RPC messages from stdin (Claude Code), POSTs them to the
 * discord-server's StreamableHTTP endpoint, parses SSE responses, and
 * writes JSON-RPC messages back to stdout. Also maintains a long-lived
 * GET SSE stream for server-initiated notifications.
 *
 * Zero dependencies — Bun builtins only.
 */

const SERVER_URL = process.env.MCP_SERVER_URL ?? 'http://127.0.0.1:9877/mcp'
const STARTUP_RETRIES = [500, 1000, 2000, 4000, 8000]
const NOTIFY_RECONNECT_BASE_MS = 1000
const NOTIFY_RECONNECT_MAX_MS = 30000

let sessionId: string | undefined
let notifyController: AbortController | undefined

function write(msg: unknown): void {
  process.stdout.write(JSON.stringify(msg) + '\n')
}

function* parseSSE(text: string): Generator<unknown> {
  for (const line of text.split('\n')) {
    if (line.startsWith('data:')) {
      const payload = line.slice(5).trim()
      if (payload) {
        try { yield JSON.parse(payload) }
        catch (e) { process.stderr.write(`SSE parse error: ${e}\n`) }
      }
    }
  }
}

async function postMessage(msg: unknown): Promise<void> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json, text/event-stream',
  }
  if (sessionId) headers['Mcp-Session-Id'] = sessionId

  let res: Response | undefined
  let lastErr: unknown

  const attempts = sessionId ? [0] : [0, ...STARTUP_RETRIES]
  for (let i = 0; i < attempts.length; i++) {
    if (i > 0) await Bun.sleep(attempts[i])
    try {
      res = await fetch(SERVER_URL, { method: 'POST', headers, body: JSON.stringify(msg) })
      break
    } catch (e) { lastErr = e }
  }

  if (!res) {
    process.stderr.write(`proxy: server unreachable for POST: ${lastErr}\n`)
    const reqMsg = msg as Record<string, unknown>
    if (reqMsg.id) {
      write({
        jsonrpc: '2.0',
        error: { code: -32000, message: 'Discord server temporarily unavailable, reconnecting...' },
        id: reqMsg.id,
      })
    }
    // Clear session — the notification loop will handle reconnection
    sessionId = undefined
    restartNotificationStream()
    return
  }

  const newSession = res.headers.get('Mcp-Session-Id')
  if (newSession && newSession !== sessionId) {
    const old = sessionId
    sessionId = newSession
    process.stderr.write(`proxy: session ${old?.slice(0, 8) ?? 'none'} -> ${newSession.slice(0, 8)}\n`)
    restartNotificationStream()
  }

  const ct = res.headers.get('Content-Type') ?? ''
  if (ct.includes('text/event-stream')) {
    const body = await res.text()
    for (const parsed of parseSSE(body)) write(parsed)
  } else if (ct.includes('application/json')) {
    write(await res.json())
  } else if (res.status === 202) {
    // no body
  } else {
    process.stderr.write(`Unexpected response: ${res.status} ${ct}\n`)
    const body = await res.text()
    if (body) process.stderr.write(body + '\n')
  }
}

/**
 * Abort any existing notification stream so the loop restarts cleanly.
 * The notification loop detects the abort and reconnects on its own.
 */
function restartNotificationStream(): void {
  if (notifyController) {
    notifyController.abort()
    // Don't clear — the running loop will see the abort and restart itself
  }
}

/**
 * Send initialize + initialized to establish a fresh MCP session.
 * Does NOT write to stdout. Returns true if session established.
 */
async function reinitialize(): Promise<boolean> {
  try {
    const initRes = await fetch(SERVER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json, text/event-stream',
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'discord-proxy', version: '1.0.0' },
        },
        id: `reinit-${Date.now()}`,
      }),
      signal: AbortSignal.timeout(10000),
    })

    const newSession = initRes.headers.get('Mcp-Session-Id')
    await initRes.text().catch(() => {})

    if (!newSession) {
      process.stderr.write(`proxy: reinit got no session ID (${initRes.status})\n`)
      return false
    }

    sessionId = newSession
    process.stderr.write(`proxy: reinit session ${newSession.slice(0, 8)}\n`)

    // Send initialized notification
    try {
      const notifRes = await fetch(SERVER_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json, text/event-stream',
          'Mcp-Session-Id': newSession,
        },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
        signal: AbortSignal.timeout(5000),
      })
      await notifRes.text().catch(() => {})
    } catch {}

    return true
  } catch (e) {
    process.stderr.write(`proxy: reinit failed: ${e}\n`)
    return false
  }
}

/**
 * Maintain a long-lived GET SSE stream for server notifications.
 * This function runs forever in the background, reconnecting as needed.
 * It is designed to be called exactly once. Aborts via notifyController
 * cause it to reconnect (not exit).
 */
async function notificationLoop(): Promise<void> {
  let delay = 0

  while (true) {
    if (delay > 0) {
      process.stderr.write(`proxy: notification stream reconnecting in ${delay}ms\n`)
      await Bun.sleep(delay)
    }

    // Ensure we have a session
    if (!sessionId) {
      process.stderr.write(`proxy: no session, attempting reinit\n`)
      if (!(await reinitialize())) {
        delay = Math.min(delay ? delay * 2 : NOTIFY_RECONNECT_BASE_MS, NOTIFY_RECONNECT_MAX_MS)
        continue
      }
    }

    // Create a fresh abort controller for this attempt
    notifyController = new AbortController()
    const currentController = notifyController
    const currentSession = sessionId!

    // Try to open the SSE stream
    let res: Response
    try {
      res = await fetch(SERVER_URL, {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream',
          'Mcp-Session-Id': currentSession,
        },
        signal: currentController.signal,
      })
    } catch (e) {
      if (currentController.signal.aborted) {
        // We were told to restart — no delay
        delay = 0
        continue
      }
      delay = Math.min(delay ? delay * 2 : NOTIFY_RECONNECT_BASE_MS, NOTIFY_RECONNECT_MAX_MS)
      process.stderr.write(`proxy: notification GET failed, retry in ${delay}ms: ${e}\n`)
      continue
    }

    if (!res.ok) {
      const errBody = await res.text().catch(() => '')
      if (currentController.signal.aborted) {
        delay = 0
        continue
      }
      process.stderr.write(`proxy: notification GET rejected (${res.status}): ${errBody.slice(0, 200)}\n`)

      // 409 = SSE stream already exists on server, try reinitializing to get a fresh session
      if (res.status === 409) {
        process.stderr.write(`proxy: SSE conflict — forcing new session\n`)
        sessionId = undefined  // Force reinitialize to get fresh session
      }

      if (await reinitialize()) {
        delay = 0
        continue
      }
      delay = Math.min(delay ? delay * 2 : NOTIFY_RECONNECT_BASE_MS, NOTIFY_RECONNECT_MAX_MS)
      continue
    }

    if (!res.body) {
      delay = NOTIFY_RECONNECT_BASE_MS
      continue
    }

    // Successfully connected
    process.stderr.write(`proxy: SSE stream connected (session ${currentSession.slice(0, 8)})\n`)
    delay = 0  // Reset backoff on successful connect

    const decoder = new TextDecoder()
    let buffer = ''
    let chunkCount = 0

    try {
      for await (const chunk of res.body) {
        chunkCount++
        const text = decoder.decode(chunk as Uint8Array, { stream: true })
        buffer += text
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const payload = line.slice(5).trim()
            if (payload) {
              try {
                const parsed = JSON.parse(payload)
                process.stderr.write(`proxy: notification -> stdout (method=${parsed?.method ?? '?'})\n`)
                write(parsed)
              }
              catch (e) { process.stderr.write(`Notification SSE parse error: ${e}\n`) }
            }
          }
        }
      }
      process.stderr.write(`proxy: SSE stream ended after ${chunkCount} chunks\n`)
    } catch (e) {
      if (currentController.signal.aborted) {
        process.stderr.write(`proxy: SSE stream aborted after ${chunkCount} chunks — reconnecting\n`)
        delay = 0
        continue
      }
      process.stderr.write(`proxy: SSE stream error after ${chunkCount} chunks: ${e}\n`)
    }

    // Stream ended naturally — reconnect after brief delay
    delay = NOTIFY_RECONNECT_BASE_MS
  }
}

async function readStdin(): Promise<void> {
  const decoder = new TextDecoder()
  let buffer = ''
  let notificationLoopStarted = false

  for await (const chunk of process.stdin) {
    buffer += decoder.decode(chunk as Uint8Array, { stream: true })

    let newlineIdx: number
    while ((newlineIdx = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newlineIdx).trim()
      buffer = buffer.slice(newlineIdx + 1)
      if (!line) continue

      let msg: unknown
      try { msg = JSON.parse(line) }
      catch (e) {
        process.stderr.write(`Invalid JSON from stdin: ${e}\n`)
        continue
      }

      await postMessage(msg)

      // Start the notification loop once we have a session
      if (!notificationLoopStarted && sessionId) {
        notificationLoopStarted = true
        notificationLoop()  // Runs forever in background (not awaited)
      }
    }
  }

  if (notifyController) notifyController.abort()
  process.exit(0)
}

readStdin()
