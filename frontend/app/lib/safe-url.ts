/**
 * Guards against attacker-influenced strings (a backend `url` field, `?next=`)
 * navigating the browser off-origin.
 *
 * Decisions use the WHATWG `URL` parser — the same parser the browser uses for
 * `href` — because browsers strip leading C0 controls/spaces and strip
 * tab/CR/LF from anywhere in a URL, so a naive prefix check is bypassable:
 * `" javascript:x"` and `"java\tscript:x"` both execute while
 * `startsWith('javascript:')` sees nothing dangerous.
 *
 * Where the guard is deliberately stricter than the browser (C0 controls,
 * NBSP/BOM, raw backslashes) it says so at the point of difference; it is
 * never looser.
 */

const SAFE_PROTOCOLS: Record<string, true> = {
  'http:': true,
  'https:': true,
}

const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+\-.]*:/

/**
 * Raw C0 controls (NUL, SOH, … US, plus DEL) anywhere in the input.
 *
 * Browsers strip leading C0 controls/spaces and strip tab/CR/LF from anywhere
 * in a URL, so `\x00//evil.com` navigates off-origin while the raw string does
 * not begin with `//` and slips past the regex below.
 */
export const CONTROL_CHAR_RE = /[\u0000-\u001F\u007F]/

/**
 * Protocol-relative references (`//host`), including the backslash spellings —
 * browsers normalise `\` to `/` for special schemes, so `\\evil.com`,
 * `/\evil.com` and `\/evil.com` are one escape. Does not catch
 * control-prefixed escapes; `CONTROL_CHAR_RE` above does, before any parsing.
 */
const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/

/**
 * Stand-in base for renders with no `window`; nothing is ever fetched from it,
 * it only lets a relative URL resolve to something so the parse below cannot
 * diverge between server and client. `.invalid` (RFC 2606, never resolvable) so
 * a stand-in origin is never a real navigation target.
 */
const SSR_BASE = 'https://ssr.invalid/'

/**
 * True when `url` is safe to place in an `href`: an absolute `http(s)://` URL
 * or a same-origin relative reference (`/path`, `path`, `?q=1`, `#frag`).
 *
 * @param base origin to resolve relative references against; defaults to the
 *             current page, falling back to a dummy origin during SSR
 */
export function isSafeUrl(url: unknown, base?: string): boolean {
  if (typeof url !== 'string') return false

  // `trim()` is deliberately kept: it also strips NBSP (U+00A0) and BOM
  // (U+FEFF), which the URL parser does not. One of three places this guard is
  // stricter than the browser (the others are CONTROL_CHAR_RE and the backslash
  // check below); all three err towards blocking.
  const raw = url.trim()
  if (!raw) return false

  // Checked before any parsing: `trim()` leaves NUL and the other C0 controls in
  // place and the parser then strips them, so a control-prefixed `\x00//host`
  // is parsed as a real protocol-relative escape.
  if (CONTROL_CHAR_RE.test(raw)) return false

  // A raw backslash is a second spelling of `/` to the parser for every special
  // scheme, so `https:/\evil.com` parses to `https://evil.com/` — an off-origin
  // host *with* an `https:` protocol, which no scheme allowlist can catch.
  // Third place this guard is stricter than the browser, and the cheapest:
  // `\` is not in the RFC 3986 URI character set, so a stored URL carrying one
  // is malformed anyway.
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

  // A reference with no scheme of its own is meant to be same-origin; if it
  // still lands on another origin it was an escape attempt, not a local path.
  if (!SCHEME_RE.test(raw) && parsed.origin !== baseUrl.origin) return false

  return true
}

/**
 * True only for a safe, same-origin, root-relative redirect path. A type guard
 * on purpose: it narrows the `URLSearchParams.get` result for callers in one
 * step.
 *
 * A leading `/` is required, which alone rules out any absolute reference. The
 * residue worth catching is control-prefixed escapes: the parser REMOVES tab,
 * CR and LF, so `/\t/evil.com` reads as a harmless local path here and as
 * `//evil.com` in the browser.
 *
 * This is the ONLY implementation: `login`, `signup` and `redirectToLogin` all
 * use it, so the open-redirect protection cannot differ between them. Rejected
 * means "the caller falls back to its default" (`/chat`, `/`), never pass-through.
 */
export function isSafeRedirect(next: unknown): next is string {
  if (typeof next !== 'string' || next.length === 0) return false

  // Raw string, not `url.trim()`: a leading space already fails the
  // `startsWith('/')` check below, and refusing control characters is the check
  // that actually matters.
  if (CONTROL_CHAR_RE.test(next)) return false

  if (!next.startsWith('/')) return false

  // Rejects `//evil.com` and every backslash spelling of the same escape.
  if (PROTOCOL_RELATIVE_RE.test(next)) return false

  // Unreachable while the `startsWith('/')` check above stands: a value starting
  // with `/` can never match a `^[a-zA-Z]` scheme prefix. Kept so loosening that
  // check cannot silently admit a scheme.
  if (SCHEME_RE.test(next)) return false

  return true
}

