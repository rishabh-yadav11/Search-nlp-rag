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
 * Raw C0 controls (NUL, SOH, … US, plus DEL) anywhere in the input.
 *
 * Browsers strip leading C0 controls and spaces, and strip tab/CR/LF from
 * anywhere in a URL, so `\x00//evil.com` navigates off-origin even though the
 * raw string does not begin with `//` and so slips past the regex above. A
 * fuzz over 5046 inputs showed that every server/client verdict divergence
 * this guard could produce was a control-prefixed reference: it resolves to
 * the stand-in base on the server but to the real page origin in the browser,
 * so the two environments disagreed — and in the dangerous direction, the
 * server marked it safe and emitted a clickable off-origin link that React
 * then discarded on hydration.
 *
 * Rejecting raw C0 controls removes that whole class. It costs nothing: no
 * legitimate URL contains them, since browsers percent-encode them anyway.
 */
const CONTROL_CHAR_RE = /[\u0000-\u001F\u007F]/

/**
 * Rejects protocol-relative references (`//host`), including the backslash
 * spellings — browsers normalise `\` to `/` for special schemes, so
 * `\\evil.com`, `/\evil.com` and `\/evil.com` are the same escape.
 *
 * This regex *is* load-bearing: deleting it (making it never match) fails the
 * parity test for `//ssr.invalid/x`, which would otherwise resolve to exactly
 * the stand-in base origin on the server and be rejected by the client. It is
 * not what catches control-prefixed escapes like `\x00//evil.com` —
 * `CONTROL_CHAR_RE` above does that, before any parsing happens.
 */
const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/

/**
 * Stand-in base for renders with no `window`. Nothing is ever fetched from it;
 * it only lets a relative URL resolve to *something* instead of throwing, so
 * the parse below does not diverge between server and client.
 *
 * The host is `.invalid` (RFC 2606, never resolvable) so a stand-in origin is
 * never a real navigation target. Stated precisely, because the distinction
 * matters: this is defence-in-depth, *not* the fix for the reported
 * divergence. Reverting it to `https://localhost/` no longer fails any test,
 * because `CONTROL_CHAR_RE` refuses the control-prefixed inputs
 * (`\x00//localhost`) before parsing on both paths. It is kept because a
 * routable stand-in would turn any future parser-level gap into a real
 * navigation to that host.
 */
const SSR_BASE = 'https://ssr.invalid/'

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

  // Checked before any parsing: `trim()` leaves NUL and the other C0 controls
  // in place, and the parser then strips them, so a control-prefixed
  // `\x00//host` is parsed as a real protocol-relative escape. Rejecting the
  // raw form is what keeps the server verdict equal to the client verdict.
  if (CONTROL_CHAR_RE.test(raw)) return false

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
  // Defence-in-depth: with CONTROL_CHAR_RE and PROTOCOL_RELATIVE_RE in front
  // of it, removing this check fails no test. It is kept because it states the
  // invariant directly and would catch a new escape shape that neither regex
  // anticipated, rather than relying on the parser to normalise it safely.
  if (!SCHEME_RE.test(raw) && parsed.origin !== baseUrl.origin) return false

  return true
}

