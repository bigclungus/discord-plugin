/**
 * Voice receive + speech-to-text module for the Discord plugin.
 *
 * Subscribes to audio from users in a voice channel, decodes Opus to PCM,
 * downsamples 48 kHz stereo to 16 kHz mono, writes a WAV file, runs Whisper
 * for transcription, and injects the result as a "[vc]" message via the
 * local inject endpoint.
 *
 * On teardown (last person leaves), compiles all transcriptions from the
 * session and uploads them as a .txt file to #voice-transcripts.
 *
 * Usage (from server.ts after consolidation):
 *   import { setupVoiceReceive } from './voice-receive'
 *   const teardown = setupVoiceReceive(connection, client, injectSecret, injectChatId)
 *   await teardown()  // returns a Promise now
 */

import { type VoiceConnection, EndBehaviorType } from '@discordjs/voice'
import { OpusEncoder } from '@discordjs/opus'
import { type Client } from 'discord.js'
import { spawn } from 'child_process'
import { writeFileSync, unlinkSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { randomBytes } from 'crypto'

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Silence timeout — flush buffer after this many ms without a packet. */
const SILENCE_TIMEOUT_MS = 1500

/** Max recording duration per user utterance (ms). */
const MAX_DURATION_MS = 15_000

/** How often we check for stale buffers (ms). */
const FLUSH_INTERVAL_MS = 500

/** Minimum word count to accept a transcription. */
const MIN_WORD_COUNT = 3

/** Whisper hallucination phrases that appear on silence / noise. */
const HALLUCINATION_PHRASES: ReadonlySet<string> = new Set([
  'thank you',
  'thanks for watching',
  'thanks for listening',
  'bye',
  'goodbye',
  'you',
  'the end',
  'thank you for watching',
  'thanks for watching!',
  'subscribe',
  'like and subscribe',
  '',
])

/** Path to the whisper transcription helper script. */
const WHISPER_SCRIPT = '/mnt/data/scripts/whisper-transcribe.py'

/** Inject endpoint for posting transcriptions back to BigClungus. */
const INJECT_URL = 'http://127.0.0.1:9876/inject'

/** Default chat ID (main channel). Caller can override. */
const DEFAULT_CHAT_ID = '1485343472952148008'

/** Channel ID for posting session transcripts. */
const TRANSCRIPT_CHANNEL_ID = '1487240658568744980'

/** Discord API base URL. */
const DISCORD_API = 'https://discord.com/api/v10'

/** Temp directory for WAV files. */
const TMP_DIR = join(tmpdir(), 'bc-voice-stt')

// ---------------------------------------------------------------------------
// Per-user audio buffer
// ---------------------------------------------------------------------------

interface UserAudioBuffer {
  /** Raw 16-bit PCM chunks (already downsampled to 16 kHz mono). */
  chunks: Buffer[]
  /** Total byte length across all chunks. */
  byteLength: number
  /** Timestamp of last received audio packet. */
  lastPacketTime: number
  /** Timestamp when recording started (first packet). */
  startTime: number
  /** Resolved Discord username. */
  username: string
}

// ---------------------------------------------------------------------------
// Session transcript
// ---------------------------------------------------------------------------

interface TranscriptEntry {
  username: string
  text: string
  timestamp: Date
  isBot: boolean
}

// ---------------------------------------------------------------------------
// Exported teardown type (teardown function with logBotSpeech attached)
// ---------------------------------------------------------------------------

export interface VoiceTeardown {
  (): Promise<void>
  logBotSpeech: (text: string) => void
}

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

/**
 * Downsample interleaved 16-bit PCM from 48 kHz stereo to 16 kHz mono.
 *
 * Strategy: average the two stereo channels, then take every 3rd sample
 * (48000 / 16000 = 3).
 */
function downsample48kStereoTo16kMono(pcm: Buffer): Buffer {
  const sampleCount = pcm.length / 4  // 2 bytes * 2 channels per sample
  const outSamples = Math.floor(sampleCount / 3)
  const out = Buffer.alloc(outSamples * 2)

  for (let i = 0; i < outSamples; i++) {
    const srcOffset = i * 3 * 4  // 3x decimation, 4 bytes per stereo sample
    if (srcOffset + 3 >= pcm.length) break
    const left = pcm.readInt16LE(srcOffset)
    const right = pcm.readInt16LE(srcOffset + 2)
    const mono = Math.round((left + right) / 2)
    out.writeInt16LE(Math.max(-32768, Math.min(32767, mono)), i * 2)
  }

  return out
}

/**
 * Build a minimal WAV header + data for 16 kHz 16-bit mono PCM.
 */
function buildWav(pcmData: Buffer): Buffer {
  const sampleRate = 16000
  const numChannels = 1
  const bitsPerSample = 16
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8)
  const blockAlign = numChannels * (bitsPerSample / 8)
  const dataSize = pcmData.length
  const headerSize = 44

  const header = Buffer.alloc(headerSize)

  // RIFF header
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + dataSize, 4)
  header.write('WAVE', 8)

  // fmt sub-chunk
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16)           // sub-chunk size
  header.writeUInt16LE(1, 20)            // PCM format
  header.writeUInt16LE(numChannels, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(byteRate, 28)
  header.writeUInt16LE(blockAlign, 32)
  header.writeUInt16LE(bitsPerSample, 34)

  // data sub-chunk
  header.write('data', 36)
  header.writeUInt32LE(dataSize, 40)

  return Buffer.concat([header, pcmData])
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

/**
 * Run whisper-transcribe.py on a WAV file and return the text.
 */
function transcribeWav(wavPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [WHISPER_SCRIPT, wavPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 60_000,
    })

    const stdoutChunks: Buffer[] = []
    const stderrChunks: Buffer[] = []

    proc.stdout.on('data', (chunk: Buffer) => stdoutChunks.push(chunk))
    proc.stderr.on('data', (chunk: Buffer) => stderrChunks.push(chunk))

    proc.on('close', (code) => {
      if (code !== 0) {
        const stderr = Buffer.concat(stderrChunks).toString().trim()
        reject(new Error(`whisper exited ${code}: ${stderr}`))
        return
      }
      resolve(Buffer.concat(stdoutChunks).toString().trim())
    })

    proc.on('error', reject)
  })
}

/**
 * Check if a transcription is valid (not a hallucination, long enough).
 */
function isValidTranscription(text: string): boolean {
  const normalized = text.toLowerCase().replace(/[.,!?]/g, '').trim()
  if (HALLUCINATION_PHRASES.has(normalized)) return false
  const wordCount = normalized.split(/\s+/).filter(Boolean).length
  return wordCount >= MIN_WORD_COUNT
}

// ---------------------------------------------------------------------------
// Inject endpoint
// ---------------------------------------------------------------------------

async function injectTranscription(
  username: string,
  text: string,
  injectSecret: string,
  chatId: string,
): Promise<void> {
  const body = JSON.stringify({
    content: `[vc] ${username}: ${text}`,
    chat_id: chatId,
    user: username,
  })

  const resp = await fetch(INJECT_URL, {
    method: 'POST',
    headers: {
      'x-inject-secret': injectSecret,
      'Content-Type': 'application/json',
    },
    body,
  })

  if (!resp.ok) {
    const detail = await resp.text().catch(() => '(no body)')
    throw new Error(`inject failed ${resp.status}: ${detail}`)
  }
}

// ---------------------------------------------------------------------------
// Discord API helper — post message to a channel
// ---------------------------------------------------------------------------

async function postToChannel(
  channelId: string,
  content: string,
  botToken: string,
): Promise<void> {
  const resp = await fetch(`${DISCORD_API}/channels/${channelId}/messages`, {
    method: 'POST',
    headers: {
      'Authorization': `Bot ${botToken}`,
      'Content-Type': 'application/json',
      'User-Agent': 'BigClungus/1.0',
    },
    body: JSON.stringify({ content }),
  })

  if (!resp.ok) {
    const detail = await resp.text().catch(() => '(no body)')
    throw new Error(`Discord POST to channel ${channelId} failed ${resp.status}: ${detail}`)
  }
}

/**
 * Upload a .txt file to a Discord channel.
 */
async function postFileToChannel(
  channelId: string,
  filename: string,
  fileContent: string,
  messageText: string,
  botToken: string,
): Promise<void> {
  const boundary = `----BigClungus${randomBytes(8).toString('hex')}`
  const fileBuf = Buffer.from(fileContent, 'utf-8')

  const parts: Buffer[] = []

  // JSON payload part
  const payloadJson = JSON.stringify({
    content: messageText,
    attachments: [{ id: 0, filename }],
  })
  parts.push(Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\nContent-Type: application/json\r\n\r\n${payloadJson}\r\n`
  ))

  // File part
  parts.push(Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="files[0]"; filename="${filename}"\r\nContent-Type: text/plain\r\n\r\n`
  ))
  parts.push(fileBuf)
  parts.push(Buffer.from(`\r\n--${boundary}--\r\n`))

  const body = Buffer.concat(parts)

  const resp = await fetch(`${DISCORD_API}/channels/${channelId}/messages`, {
    method: 'POST',
    headers: {
      'Authorization': `Bot ${botToken}`,
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'User-Agent': 'BigClungus/1.0',
    },
    body,
  })

  if (!resp.ok) {
    const detail = await resp.text().catch(() => '(no body)')
    throw new Error(`Discord file upload to ${channelId} failed ${resp.status}: ${detail}`)
  }
}

// ---------------------------------------------------------------------------
// Transcript formatting
// ---------------------------------------------------------------------------

function formatTimestamp(date: Date): string {
  const h = date.getHours().toString().padStart(2, '0')
  const m = date.getMinutes().toString().padStart(2, '0')
  const s = date.getSeconds().toString().padStart(2, '0')
  return `${h}:${m}:${s}`
}

function formatTranscript(entries: TranscriptEntry[], sessionStart: Date): string {
  const now = new Date()
  const durationMs = now.getTime() - sessionStart.getTime()
  const durationMin = Math.round(durationMs / 60_000)

  const dateStr = sessionStart.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  const header = `**VC Session Transcript** \u2014 ${dateStr}\nDuration: ${durationMin} minute${durationMin !== 1 ? 's' : ''}\n\n`

  const lines = entries.map((e) => {
    const ts = formatTimestamp(e.timestamp)
    const botMarker = e.isBot ? ' \uD83D\uDD0A' : ''
    return `[${ts}] **${e.username}**${botMarker}: ${e.text}`
  })

  return header + lines.join('\n')
}

// ---------------------------------------------------------------------------
// Username resolution
// ---------------------------------------------------------------------------

/** Cache user ID -> username to avoid repeated API calls. */
const usernameCache = new Map<string, string>()

async function resolveUsername(client: Client, userId: string): Promise<string> {
  const cached = usernameCache.get(userId)
  if (cached) return cached

  try {
    const user = await client.users.fetch(userId)
    const name = user.displayName || user.username
    usernameCache.set(userId, name)
    return name
  } catch {
    return `user-${userId.slice(-4)}`
  }
}

// ---------------------------------------------------------------------------
// Core: setupVoiceReceive
// ---------------------------------------------------------------------------

/**
 * Attach audio receive handlers to a VoiceConnection.
 *
 * IMPORTANT: The connection must have been created with `selfDeaf: false`
 * for the receiver to get audio packets.
 *
 * @param connection   - An active, ready VoiceConnection
 * @param client       - The Discord.js Client (for resolving usernames)
 * @param injectSecret - The DISCORD_INJECT_SECRET value
 * @param injectChatId - Chat ID to inject transcriptions into (defaults to main channel)
 * @returns A teardown function (async) that stops all listening, flushes pending
 *          transcriptions, and posts the session transcript. Has a .logBotSpeech()
 *          method for recording bot speech events.
 */
export function setupVoiceReceive(
  connection: VoiceConnection,
  client: Client,
  injectSecret: string,
  injectChatId: string = DEFAULT_CHAT_ID,
): VoiceTeardown {
  // Ensure temp dir exists
  mkdirSync(TMP_DIR, { recursive: true })

  const buffers = new Map<string, UserAudioBuffer>()
  const decoder = new OpusEncoder(48000, 2)  // Discord sends 48 kHz stereo Opus
  let destroyed = false

  // Session transcript accumulator
  const transcript: TranscriptEntry[] = []
  const sessionStart = new Date()

  // -----------------------------------------------------------------------
  // Flush a single user's buffer: downsample, write WAV, transcribe, inject.
  // -----------------------------------------------------------------------
  async function flushBuffer(userId: string): Promise<void> {
    const buf = buffers.get(userId)
    if (!buf || buf.chunks.length === 0) return
    buffers.delete(userId)

    // Concatenate all PCM chunks (already 16 kHz mono)
    const pcm = Buffer.concat(buf.chunks)
    const durationSec = (pcm.length / (16000 * 2)).toFixed(2)  // 16kHz 16-bit mono = 32000 bytes/sec

    console.log(`[vc-debug] flushBuffer userId=${userId} username=${buf.username} chunks=${buf.chunks.length} bytes=${pcm.length} duration=${durationSec}s`)

    // Skip very short audio (< 0.5s at 16kHz 16-bit mono = 16000 bytes)
    if (pcm.length < 16000) {
      console.log(`[vc-debug] flushBuffer FILTERED: too short (${pcm.length} bytes < 16000) for ${buf.username}`)
      return
    }

    const wavPath = join(TMP_DIR, `vc-${randomBytes(6).toString('hex')}.wav`)

    try {
      const wav = buildWav(pcm)
      writeFileSync(wavPath, wav)

      const text = await transcribeWav(wavPath)

      if (isValidTranscription(text)) {
        console.log(`[vc-debug] transcription ACCEPTED for ${buf.username}: "${text}" — injecting to chatId=${injectChatId}`)
        await injectTranscription(buf.username, text, injectSecret, injectChatId)

        // Add to session transcript
        transcript.push({
          username: buf.username,
          text,
          timestamp: new Date(),
          isBot: false,
        })
      } else {
        // Log why it was filtered
        const normalized = text.toLowerCase().replace(/[.,!?]/g, '').trim()
        const isHallucination = HALLUCINATION_PHRASES.has(normalized)
        const wordCount = normalized.split(/\s+/).filter(Boolean).length
        console.log(`[vc-debug] transcription FILTERED for ${buf.username}: "${text}" — hallucination=${isHallucination} wordCount=${wordCount} minRequired=${MIN_WORD_COUNT}`)
      }
    } catch (err) {
      process.stderr.write(
        `voice-receive: transcription error for ${buf.username}: ${err instanceof Error ? err.message : String(err)}\n`,
      )
    } finally {
      try {
        unlinkSync(wavPath)
      } catch {
        // file may not exist if write failed
      }
    }
  }

  // -----------------------------------------------------------------------
  // Handle a user starting to speak: subscribe to their audio stream.
  // -----------------------------------------------------------------------
  function onSpeakingStart(userId: string): void {
    if (destroyed) return

    // If there's already a subscription for this user, don't double-subscribe.
    if (connection.receiver.subscriptions.has(userId)) {
      console.log(`[vc-debug] onSpeakingStart userId=${userId} — already subscribed, skipping`)
      return
    }

    console.log(`[vc-debug] onSpeakingStart userId=${userId} — subscribing to audio stream`)
    resolveUsername(client, userId).then((name) => {
      console.log(`[vc-debug] onSpeakingStart resolved username=${name} for userId=${userId}`)
    }).catch(() => {})

    const stream = connection.receiver.subscribe(userId, {
      end: {
        behavior: EndBehaviorType.Manual,
      },
    })

    // Resolve username in background (cached after first call)
    const usernamePromise = resolveUsername(client, userId)

    stream.on('data', (opusPacket: Buffer) => {
      if (destroyed) return

      const now = Date.now()

      let userBuf = buffers.get(userId)
      if (!userBuf) {
        userBuf = {
          chunks: [],
          byteLength: 0,
          lastPacketTime: now,
          startTime: now,
          username: `user-${userId.slice(-4)}`,  // placeholder until resolved
        }
        buffers.set(userId, userBuf)

        // Fill in the real username once resolved
        usernamePromise.then((name) => {
          const current = buffers.get(userId)
          if (current) current.username = name
        }).catch(() => {})
      }

      userBuf.lastPacketTime = now

      // Decode Opus -> 48 kHz stereo PCM, then downsample to 16 kHz mono
      try {
        const pcm48kStereo = decoder.decode(opusPacket)
        const pcm16kMono = downsample48kStereoTo16kMono(pcm48kStereo)
        userBuf.chunks.push(pcm16kMono)
        userBuf.byteLength += pcm16kMono.length
      } catch (err) {
        // Malformed Opus packet -- skip
        process.stderr.write(
          `voice-receive: opus decode error: ${err instanceof Error ? err.message : String(err)}\n`,
        )
      }

      // Check max duration
      if (now - userBuf.startTime >= MAX_DURATION_MS) {
        // Destroy the stream so we stop receiving, then flush
        stream.destroy()
        connection.receiver.subscriptions.delete(userId)
        flushBuffer(userId).catch(() => {})
      }
    })

    stream.on('end', () => {
      connection.receiver.subscriptions.delete(userId)
    })

    stream.on('error', (err: Error) => {
      process.stderr.write(`voice-receive: stream error for ${userId}: ${err.message}\n`)
      connection.receiver.subscriptions.delete(userId)
    })
  }

  // Subscribe to speaking events
  connection.receiver.speaking.on('start', onSpeakingStart)

  // -----------------------------------------------------------------------
  // Periodic flush: detect silence (no packets for SILENCE_TIMEOUT_MS).
  // -----------------------------------------------------------------------
  const flushInterval = setInterval(() => {
    if (destroyed) return

    const now = Date.now()
    for (const [userId, buf] of buffers) {
      const silenceMs = now - buf.lastPacketTime
      if (silenceMs >= SILENCE_TIMEOUT_MS) {
        // Destroy the subscription stream if still active
        const sub = connection.receiver.subscriptions.get(userId)
        if (sub) {
          sub.destroy()
          connection.receiver.subscriptions.delete(userId)
        }
        flushBuffer(userId).catch(() => {})
      }
    }
  }, FLUSH_INTERVAL_MS)

  // -----------------------------------------------------------------------
  // Teardown function (async — awaits pending flushes + posts transcript)
  // -----------------------------------------------------------------------
  async function teardown(): Promise<void> {
    destroyed = true
    clearInterval(flushInterval)
    connection.receiver.speaking.removeListener('start', onSpeakingStart)

    // Destroy all active subscriptions
    for (const [userId, sub] of connection.receiver.subscriptions) {
      sub.destroy()
      connection.receiver.subscriptions.delete(userId)
    }

    // Flush any remaining buffers and await them all
    const flushPromises: Promise<void>[] = []
    for (const [userId] of buffers) {
      flushPromises.push(flushBuffer(userId))
    }
    await Promise.allSettled(flushPromises)

    // Post session transcript if we have entries — always as a .txt file upload
    if (transcript.length > 0) {
      const botToken = process.env.DISCORD_BOT_TOKEN
      if (!botToken) {
        process.stderr.write('voice-receive: cannot post transcript — no DISCORD_BOT_TOKEN\n')
        return
      }

      try {
        const fullText = formatTranscript(transcript, sessionStart)

        const dateSlug = sessionStart.toISOString().replace(/[:.]/g, '-').slice(0, 19)
        const filename = `vc-transcript-${dateSlug}.txt`

        // Plain text version for the file (strip markdown bold)
        const plainText = fullText.replace(/\*\*/g, '')

        const durationMs = Date.now() - sessionStart.getTime()
        const durationMin = Math.round(durationMs / 60_000)
        const dateStr = sessionStart.toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric',
        })
        const summary = `VC Session \u2014 ${dateStr} \u2014 ${durationMin} min, ${transcript.length} messages`

        await postFileToChannel(TRANSCRIPT_CHANNEL_ID, filename, plainText, summary, botToken)

        process.stderr.write(
          `voice-receive: posted session transcript (${transcript.length} entries) to #voice-transcripts\n`,
        )
      } catch (err) {
        process.stderr.write(
          `voice-receive: failed to post transcript: ${err instanceof Error ? err.message : String(err)}\n`,
        )
      }
    }

    // Clear transcript
    transcript.length = 0
  }

  // Build the VoiceTeardown: teardown function with logBotSpeech attached
  const result = teardown as VoiceTeardown
  result.logBotSpeech = (text: string) => {
    transcript.push({
      username: 'BigClungus',
      text,
      timestamp: new Date(),
      isBot: true,
    })
  }

  return result
}
