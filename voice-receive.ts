/**
 * Voice receive + speech-to-text module for the Discord plugin.
 *
 * Subscribes to audio from users in a voice channel, decodes Opus to PCM,
 * downsamples 48 kHz stereo to 16 kHz mono, writes a WAV file, runs Whisper
 * for transcription, and injects the result as a "[vc]" message via the
 * local inject endpoint.
 *
 * Usage (from server.ts after consolidation):
 *   import { setupVoiceReceive } from './voice-receive'
 *   setupVoiceReceive(connection, client, injectSecret, injectChatId)
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
// Username resolution
// ---------------------------------------------------------------------------

/** Cache user ID → username to avoid repeated API calls. */
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
 * @returns A teardown function that stops all listening and cleans up.
 */
export function setupVoiceReceive(
  connection: VoiceConnection,
  client: Client,
  injectSecret: string,
  injectChatId: string = DEFAULT_CHAT_ID,
): () => void {
  // Ensure temp dir exists
  mkdirSync(TMP_DIR, { recursive: true })

  const buffers = new Map<string, UserAudioBuffer>()
  const decoder = new OpusEncoder(48000, 2)  // Discord sends 48 kHz stereo Opus
  let destroyed = false

  // -----------------------------------------------------------------------
  // Flush a single user's buffer: downsample, write WAV, transcribe, inject.
  // -----------------------------------------------------------------------
  async function flushBuffer(userId: string): Promise<void> {
    const buf = buffers.get(userId)
    if (!buf || buf.chunks.length === 0) return
    buffers.delete(userId)

    // Concatenate all PCM chunks (already 16 kHz mono)
    const pcm = Buffer.concat(buf.chunks)

    // Skip very short audio (< 0.5s at 16kHz 16-bit mono = 16000 bytes)
    if (pcm.length < 16000) return

    const wavPath = join(TMP_DIR, `vc-${randomBytes(6).toString('hex')}.wav`)

    try {
      const wav = buildWav(pcm)
      writeFileSync(wavPath, wav)

      const text = await transcribeWav(wavPath)

      if (isValidTranscription(text)) {
        await injectTranscription(buf.username, text, injectSecret, injectChatId)
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
    if (connection.receiver.subscriptions.has(userId)) return

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

      // Decode Opus → 48 kHz stereo PCM, then downsample to 16 kHz mono
      try {
        const pcm48kStereo = decoder.decode(opusPacket)
        const pcm16kMono = downsample48kStereoTo16kMono(pcm48kStereo)
        userBuf.chunks.push(pcm16kMono)
        userBuf.byteLength += pcm16kMono.length
      } catch (err) {
        // Malformed Opus packet — skip
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
  // Teardown function
  // -----------------------------------------------------------------------
  return function teardown(): void {
    destroyed = true
    clearInterval(flushInterval)
    connection.receiver.speaking.removeListener('start', onSpeakingStart)

    // Destroy all active subscriptions
    for (const [userId, sub] of connection.receiver.subscriptions) {
      sub.destroy()
      connection.receiver.subscriptions.delete(userId)
    }

    // Flush any remaining buffers synchronously-ish (fire and forget)
    for (const [userId] of buffers) {
      flushBuffer(userId).catch(() => {})
    }
  }
}
