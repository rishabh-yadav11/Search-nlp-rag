// Client-only analytics session helpers.
//
// The session id is a plain (non-HttpOnly) cookie `vccircle_sid` set by JS —
// it is an analytics join key, not a security credential, so it never carries
// Secure and never needs server-side generation. It is sent as an
// `X-Session-Id` header on /search calls and as `session_id` on the click and
// interaction beacons so the backend can join a search→click→interaction
// session server-side.

const COOKIE_NAME = 'vccircle_sid'
const MAX_AGE_S = 7 * 24 * 60 * 60 // 7 days
// Backend's own bound (backend/app/main.py MAX_DWELL_TIME_MS): the cap the
// client clamps dwell to before sending, so a long-read tab cannot overflow the
// server validation. Origin: backend/app/main.py. Value mirrored, not derived.
export const MAX_DWELL_TIME_MS = 24 * 60 * 60 * 1000

// UUID v4 shape (lower/upper hex + dashes). Anything else in the cookie is a
// stale/fabricated value and is regenerated rather than trusted.
const SESSION_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

/** Safe inner-probe of the document cookie, or null when the browser is
 *  unavailable. Never throws: cookie reads are parsing, not trusted storage.
 *  Returns null when the cookie is absent OR malformed. */
export function readSessionId(): string | null {
  if (typeof document === 'undefined') return null
  let raw: string | null = null
  try {
    const match = document.cookie.match(new RegExp(`(?:^|; )${COOKIE_NAME}=([^;]*)`))
    raw = match?.[1] || null
    if (raw && SESSION_ID_RE.test(raw)) return raw
  } catch {
    return null
  }
  // Present but malformed: clear it so `ensureSessionId` mints a fresh one
  // (writing an empty cookie removes it under any path prefix).
  if (raw) {
    try {
      document.cookie = `${COOKIE_NAME}=; path=/; max-age=0`
    } catch {
      /* cookie unreadable — return null anyway */
    }
  }
  return null
}

function newId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    // crypto.randomUUID is unavailable in insecure non-local contexts; build a
    // well-formed v4-compatible fallback from the browser PRNG bytes instead.
    const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('')
    const bytes = new Uint8Array(16)
    crypto.getRandomValues(bytes)
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const h = hex(bytes)
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
  }
}

/** The current session id, creating + persisting the cookie on first call and
 *  repairing a malformed/stale value. Never throws. */
export function ensureSessionId(): string {
  const existing = readSessionId()
  if (existing) return existing
  let id = ''
  try {
    id = newId()
  } catch {
    /* document unavailable — return an ephemeral value */ return ''
  }
  try {
    document.cookie = `${COOKIE_NAME}=${id}; path=/; max-age=${MAX_AGE_S}; samesite=lax`
  } catch {
    /* cookie write failed — return it anyway; beacon can retry */
  }
  return id
}

/** Cap measured dwell (ms) at the backend bound; non-positive/non-finite
 *  measurements collapse to 0 (meaning "not measured") instead of an outlier. */
export function clampDwell(elapsedMs: number): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) return 0
  return Math.min(Math.round(elapsedMs), MAX_DWELL_TIME_MS)
}
