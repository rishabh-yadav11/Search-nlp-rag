








const COOKIE_NAME = 'vccircle_sid'
const MAX_AGE_S = 7 * 24 * 60 * 60 



export const MAX_DWELL_TIME_MS = 24 * 60 * 60 * 1000



const SESSION_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i


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
  
  
  if (raw) {
    try {
      document.cookie = `${COOKIE_NAME}=; path=/; max-age=0`
    } catch {}
  }
  return null
}

function newId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    
    
    const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('')
    const bytes = new Uint8Array(16)
    crypto.getRandomValues(bytes)
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    const h = hex(bytes)
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
  }
}


export function ensureSessionId(): string {
  const existing = readSessionId()
  if (existing) return existing
  let id = ''
  try {
    id = newId()
  } catch {
     return ''
  }
  try {
    document.cookie = `${COOKIE_NAME}=${id}; path=/; max-age=${MAX_AGE_S}; samesite=lax`
  } catch {}
  return id
}


export function clampDwell(elapsedMs: number): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) return 0
  return Math.min(Math.round(elapsedMs), MAX_DWELL_TIME_MS)
}
