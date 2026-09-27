/**
 * Centralised guard for URLs that come from the backend (article `url`,
 * retrieval source `url`, …) and end up in an `href`.
 *
 * Threat model: a poisoned or attacker-influenced record can put an arbitrary
 * string in `url`. Rendering that string in an `href` lets
 * `javascript:` / `data:` execute on click and `//host` (or its backslash
 * spellings) silently navigate off-origin for phishing.
 *
 * The decision is made with the WHATWG `URL` parser — the exact parser the
 * browser uses for `href` — so "what we blocked" and "what the browser does"
 * cannot disagree. A raw-string prefix check cannot do that: browsers strip
 * leading C0 controls and spaces, and strip tab/CR/LF from anywhere in the URL,
 * so `" javascript:x"`, `"\x01javascript:x"` and `"java\tscript:x"` all execute
 * while a naive `startsWith('javascript:')` sees nothing dangerous.
 */

/** Only these two schemes may ever reach an `href`. */
const SAFE_PROTOCOLS: Record<string, true> = {
  'http:': true,
  'https:': true,
}

/** `scheme:` prefix per RFC 3986, e.g. `https:`, `javascript:`, `data:`. */
const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+\-.]*:/

/**
 * Two or more leading slashes/backslashes: protocol-relative (`//evil.com`).
 * Browsers normalise `\` to `/` for special schemes, so `\\evil.com`,
 * `/\evil.com` and `\/evil.com` are the same escape and must be rejected too.
 */
const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/

/**
 * Stand-in base for renders with no `window`. Nothing is ever fetched from it;
 * it only lets a relative URL resolve to *something* instead of throwing, so
 * the origin comparison below stays consistent between server and client.
 */
const SSR_BASE = 'https://localhost/'

/**
 * True when `url` is safe to place in an `href`.
 *
 * Accepts absolute `http(s)://` URLs and same-origin relative references
 * (`/path`, `path`, `?q=1`, `#frag`). Rejects every other scheme, and rejects
 * relative references that resolve off-origin.
 *
 * @param url  the raw, untrusted URL string
 * @param base origin to resolve relative references against; defaults to the
 *             current page, falling back to a dummy origin during SSR
 */
export function isSafeUrl(url: unknown, base?: string): boolean {
  if (typeof url !== 'string') return false

  // `trim()` is deliberately left in place. It also strips NBSP (U+00A0) and
  // BOM (U+FEFF), which the URL parser does *not* strip, so a NBSP-prefixed
  // `javascript:` is blocked here but would be a harmless relative path in the
  // browser. That is the one place this guard is stricter than the browser; it
  // errs towards blocking, and loosening `trim()` would reintroduce the risk.
  const raw = url.trim()
  if (!raw) return false

  // Checked on the trimmed string, but the origin comparison below is what
  // actually catches control-character-prefixed variants such as
  // `"\x01//evil.com"`, which `String.prototype.trim` does not remove.
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

  // A reference with no scheme of its own is meant to be same-origin. If it
  // still lands on another origin it was an escape attempt, not a local path.
  if (!SCHEME_RE.test(raw) && parsed.origin !== baseUrl.origin) return false

  return true
}

