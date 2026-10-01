import { afterEach, describe, expect, it, vi } from 'vitest'
import { isSafeUrl } from './safe-url'

// A fixed base keeps these tests independent of the jsdom document origin.
const BASE = 'https://app.example.com/articles'

describe('isSafeUrl — blocks script-executing schemes', () => {
  const PAYLOADS = [
    "javascript:fetch(localStorage.getItem('vccircle_auth_token'))",
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    'JAVASCRIPT:alert(document.cookie)',
    'java\tscript:alert(1)',
    'java\nscript:alert(1)',
    'java\rscript:alert(1)',
    ' javascript:alert(1)',
    '\tjavascript:alert(1)',
    '\njavascript:alert(1)',
    '\x00javascript:alert(1)',
    '\x01javascript:alert(1)',
    '\x0bjavascript:alert(1)',
    '  \r\n\t javascript:alert(1)',
    'vbscript:msgbox(1)',
    'data:text/html,<script>alert(1)</script>',
    'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==',
    'DATA:text/html,<h1>x</h1>',
    'file:///etc/passwd',
    'blob:https://app.example.com/abc',
  ]

  it.each(PAYLOADS)('rejects %j', (payload) => {
    expect(isSafeUrl(payload, BASE)).toBe(false)
  })
})

describe('isSafeUrl — blocks protocol-relative and backslash escapes', () => {
  const ESCAPES = [
    '//evil/phish',
    '//evil.com',
    '\\\\evil.com',
    '/\\evil.com',
    '\\/evil.com',
    '\x01//evil.com',
    '  //evil.com',
    '//',
  ]

  it.each(ESCAPES)('rejects %j', (payload) => {
    expect(isSafeUrl(payload, BASE)).toBe(false)
  })
})

// The payload tables above are the guard's security contract. Unlike them, these carry a real
// scheme, so the scheme allowlist alone waves them through: the WHATWG parser treats a backslash
// as a separator, leaving a well-formed off-origin https URL. The `https://`-prefixed spellings
// are the sharp end — they match a plain `/^https?:\/\//` prefix check.
describe('isSafeUrl — blocks scheme-prefixed backslash escapes', () => {
  const SCHEME_ESCAPES = [
    'https:/\\evil.com',
    'https:\\\\evil.com',
    'https:\\/evil.com',
    'http:/\\evil.com',
    'HTTPS:/\\evil.com',
    'https:\\\\evil.com/path?q=1',
    'https://\\evil.com',
    'https://\\/evil.com',
    'https://\\\\evil.com',
  ]

  it.each(SCHEME_ESCAPES)('rejects %j', (payload) => {
    expect(isSafeUrl(payload, BASE)).toBe(false)
  })

  it.each(SCHEME_ESCAPES)('rejects %j on the server and on the client alike', (payload) => {
    vi.stubGlobal('window', undefined)
    expect(isSafeUrl(payload)).toBe(false)
    vi.stubGlobal('window', { location: { href: 'https://app.example.com/chat' } })
    expect(isSafeUrl(payload)).toBe(false)
  })

  it('resolves those payloads to an off-origin https URL in a real parser', () => {
    expect(new URL('https:/\\evil.com', BASE).href).toBe('https://evil.com/')
    expect(new URL('https://\\evil.com', BASE).href).toBe('https://evil.com/')
  })

  it('refuses the spellings the old inline https-prefix check admitted', () => {
    const OLD_INLINE_RE = /^https?:\/\//
    const ADMITTED = ['https://\\evil.com', 'https://\\/evil.com', 'https://\\\\evil.com']
    for (const payload of ADMITTED) {
      expect(OLD_INLINE_RE.test(payload)).toBe(true)
      expect(isSafeUrl(payload, BASE)).toBe(false)
    }
  })
})

describe('isSafeUrl — blocks malformed and non-string input', () => {
  const REJECTED = ['', '   ', 'http:', 'https://', undefined, null, 42, {}, ['https://ok.com']]

  it.each(REJECTED)('rejects %j', (value) => {
    expect(isSafeUrl(value, BASE)).toBe(false)
  })
})

describe('isSafeUrl — allows legitimate URLs', () => {
  const SAFE = [
    'https://vccircle.com/news/deal-123',
    'http://vccircle.com/news',
    'https://vccircle.com/news?q=1#frag',
    '/local/article/9',
    'relative/path',
    '?q=1',
    '#section',
    'HTTPS://VCCIRCLE.COM/News',
  ]

  it.each(SAFE)('allows %j', (url) => {
    expect(isSafeUrl(url, BASE)).toBe(true)
  })
})

describe('isSafeUrl — relative URLs stay same-origin', () => {
  it('allows a relative path against the given base', () => {
    expect(isSafeUrl('/articles/9', BASE)).toBe(true)
  })

  it('rejects a relative path when the base itself is unparseable', () => {
    expect(isSafeUrl('/articles/9', 'not a url')).toBe(false)
  })
})

describe('isSafeUrl — SSR parity', () => {
  // jsdom always defines `window`, so the server path must be forced or these assertions prove nothing.
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses the stand-in base when no window is available (server render)', () => {
    vi.stubGlobal('window', undefined)
    expect(typeof window).toBe('undefined')

    expect(isSafeUrl('/articles/9')).toBe(true)
  })

  it('still rejects a javascript: URL when no window is available', () => {
    vi.stubGlobal('window', undefined)
    expect(isSafeUrl('javascript:alert(1)')).toBe(false)
  })

  it('still blocks backslash escapes when no window is available', () => {
    vi.stubGlobal('window', undefined)
    expect(isSafeUrl('/\\evil.com')).toBe(false)
  })
})

describe('isSafeUrl — server/client parity', () => {
  // The guard runs in SSR and again after hydration; if the two disagree, the server emits markup the
  // client throws away — or a clickable off-origin link. The defect IS the divergence.
  const CLIENT_ORIGIN = 'https://app.vccircle.com/'

  function verdictIn(url: string, client: boolean): boolean {
    vi.stubGlobal('window', client ? { location: { href: CLIENT_ORIGIN } } : undefined)
    // Prove the stub landed: a silently-ineffective `vi.stubGlobal` would skip the server branch.
    expect(typeof window === 'undefined' ? 'server' : 'client').toBe(
      client ? 'client' : 'server'
    )
    return isSafeUrl(url)
  }

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  // `\x00` survives `String.prototype.trim`, so only the origin comparison catches these.
  const NUL_ESCAPES = ['\x00//localhost', '\x00//localhost/x', '\x00//localhost:443']

  it.each(NUL_ESCAPES)('rejects %j on the server and on the client alike', (url) => {
    expect(verdictIn(url, false)).toBe(false)
    expect(verdictIn(url, true)).toBe(false)
  })

  const CORPUS = [
    '//localhost',
    '\x00//evil.com',
    '\x00//localhost',
    '\x00//localhost:443',
    '\x00//ssr.invalid',
    '\x00//app.vccircle.com',
    '\x00/\\evil.com',
    '\x00\\/ssr.invalid',
    '\x01//evil.com',
    '\x1f//evil.com',
    '\x00javascript:alert(1)',
    '\x00data:text/html,x',
    '//ssr.invalid/x',
    '\\ssr.invalid/x',
    '/\\ssr.invalid/x',
    '\\/ssr.invalid/x',
    '\\\\ssr.invalid/x',
    '/\\evil.com',
    '\\/evil.com',
    '\\\\evil.com',
    '/rel',
    'rel/path',
    '?q=1',
    '#f',
    '',
    '   ',
    'https://ok.com/a',
    'http://ok.com/a',
    'https://ssr.invalid/x',
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    'java\tscript:alert(1)',
    ' javascript:alert(1)',
    'data:text/html,<script>alert(1)</script>',
  ]

  // Browsers strip leading C0 controls before parsing, so these look schemeless on the raw string yet
  // resolve as real protocol-relative escapes. Each must be refused outright rather than left to parsing.
  it.each(['\x00//evil.com', '\x00//ssr.invalid', '\x01//evil.com', '\x1f//localhost'])(
    'rejects the control-prefixed escape %j before parsing',
    (url) => {
      expect(verdictIn(url, false)).toBe(false)
      expect(verdictIn(url, true)).toBe(false)
    }
  )

  it.each(CORPUS)('returns the same verdict for %j on the server and the client', (url) => {
    expect(verdictIn(url, false)).toBe(verdictIn(url, true))
  })

  // Aimed at the stand-in origin, these resolve to exactly that origin on the server, so only the
  // origin comparison would wave them through there; a parity assertion compares verdicts and would
  // not catch it. The explicit `false` pins the outcome.
  it.each(['/\\ssr.invalid/x', '\\/ssr.invalid/x', '\\\\ssr.invalid/x'])(
    'rejects the backslash escape %j aimed at the stand-in origin',
    (url) => {
      expect(verdictIn(url, false)).toBe(false)
      expect(verdictIn(url, true)).toBe(false)
    }
  )

  // A browser same-origin path, not an escape — refused anyway by the backslash check, which is why
  // it needs its own pin: the CORPUS parity assertion above passes either way.
  it('refuses the same-origin single-backslash path on both sides', () => {
    expect(verdictIn('\\ssr.invalid/x', false)).toBe(false)
    expect(verdictIn('\\ssr.invalid/x', true)).toBe(false)
  })
})

describe('isSafeUrl — deliberate strictness on control characters', () => {
  // The browser percent-encodes NUL and the other C0 controls but silently REMOVES tab/CR/LF. Refusing
  // tab costs something (a real URL with a stray tab renders as inert text) and is an accepted
  // fail-closed trade-off. Relaxing CONTROL_CHAR_RE should break this block.
  it.each([
    ['NUL is percent-encoded by the browser but still refused', 'https://ok.com/a\x00b'],
    ['tab is removed by the browser but still refused', 'https://ok.com/a\tb'],
    ['CR is removed by the browser but still refused', 'https://ok.com/a\rb'],
    ['LF is removed by the browser but still refused', 'https://ok.com/a\nb'],
  ])('%s', (_label, url) => {
    expect(isSafeUrl(url, BASE)).toBe(false)
  })

  it('still allows an ordinary URL with a percent-encoded space', () => {
    // A literal space is not a C0 control: the parser encodes it and the link stays usable.
    expect(isSafeUrl('https://ok.com/a b', BASE)).toBe(true)
  })
})
