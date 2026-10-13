

const SAFE_PROTOCOLS: Record<string, true> = {
  'http:': true,
  'https:': true,
}

const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+\-.]*:/


export const CONTROL_CHAR_RE = /[\u0000-\u001F\u007F]/


const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/


const SSR_BASE = 'https://ssr.invalid/'


export function isSafeUrl(url: unknown, base?: string): boolean {
  if (typeof url !== 'string') return false

  
  
  
  
  const raw = url.trim()
  if (!raw) return false

  
  
  
  if (CONTROL_CHAR_RE.test(raw)) return false

  
  
  
  
  
  
  if (raw.includes('\\')) return false

  if (PROTOCOL_RELATIVE_RE.test(raw)) return false

  let baseUrl: URL
  try {
    baseUrl = new URL(base ?? (typeof window !== 'undefined' ? window.location.href : SSR_BASE))
  } catch {
    return false
  }

  let parsed: URL
  try {
    parsed = new URL(raw, baseUrl)
  } catch {
    return false
  }

  if (SAFE_PROTOCOLS[parsed.protocol] !== true) return false

  
  
  if (!SCHEME_RE.test(raw) && parsed.origin !== baseUrl.origin) return false

  return true
}


export function isSafeRedirect(next: unknown): next is string {
  if (typeof next !== 'string' || next.length === 0) return false

  
  
  
  if (CONTROL_CHAR_RE.test(next)) return false

  if (!next.startsWith('/')) return false

  
  if (PROTOCOL_RELATIVE_RE.test(next)) return false

  
  
  
  if (SCHEME_RE.test(next)) return false

  return true
}

