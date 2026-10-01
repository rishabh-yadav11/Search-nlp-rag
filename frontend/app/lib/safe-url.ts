/**
 * Verdicts go through the WHATWG `URL` parser — the parser the browser itself applies to
 * `href` — so our blocked set cannot disagree with what the browser would do. Never looser.
 */

/** Only these two schemes may ever reach an `href`. */
const SAFE_PROTOCOLS: Record<string, true> = {
  'http:': true,
  'https:': true,
}

/** Leading `scheme:` prefix per RFC 3986; its absence marks a relative reference. */
const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+\-.]*:/

/**
 * Raw C0 controls and DEL anywhere in the input: browsers strip leading ones and strip
 * tab/CR/LF everywhere, so the raw form is what decides and it fails closed.
 */
export const CONTROL_CHAR_RE = /[\u0000-\u001F\u007F]/

/**
 * `//host` navigates off-origin for phishing, and the browser folds `\` into `/`, so every
 * backslash spelling is that same escape.
 */
const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/

/** RFC 2606 `.invalid` stand-in origin so a relative reference still parses during SSR and can never be fetched. */
const SSR_BASE = 'https://ssr.invalid/'

/**
 * True when `url` may go in an `href`; a false verdict means the caller must render the
 * content inert, not merely leave it un-clickable.
 */
export function isSafeUrl(url: unknown, base?: string): boolean {
  if (typeof url !== 'string') return false

  // `trim()` strips NBSP/BOM, which the URL parser does not — stricter than the browser on purpose.
  const raw = url.trim()
  if (!raw) return false

  if (CONTROL_CHAR_RE.test(raw)) return false

  // For special schemes the parser reads `\` as `/`, so `https:/\evil.com` parses as an
  // off-origin `https:` URL — exactly what SAFE_PROTOCOLS admits, so no scheme check catches it.
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

  // A schemeless reference is meant to stay same-origin; this states it directly, so a new
  // escape shape neither regex anticipates is still caught.
  if (!SCHEME_RE.test(raw) && parsed.origin !== baseUrl.origin) return false

  return true
}

/**
 * Narrows on purpose — `next` is both the test and the value callers go on to use — and accepts
 * only paths starting with exactly one `/`, which is what rules out `//evil.com`, `/\evil.com` and
 * every absolute reference. A false verdict means substitute the caller's default, never pass on.
 */
export function isSafeRedirect(next: unknown): next is string {
  if (typeof next !== 'string' || next.length === 0) return false

  if (CONTROL_CHAR_RE.test(next)) return false

  if (!next.startsWith('/')) return false

  if (PROTOCOL_RELATIVE_RE.test(next)) return false

  // Unreachable while the `startsWith('/')` above stands; kept so loosening it cannot silently
  // admit a scheme.
  if (SCHEME_RE.test(next)) return false

  return true
}

