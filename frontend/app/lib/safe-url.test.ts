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
  // jsdom always defines `window`, so the server path has to be forced or
  // these assertions silently take the client branch and prove nothing.
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses the stand-in base when no window is available (server render)', () => {
    vi.stubGlobal('window', undefined)
    expect(typeof window).toBe('undefined')

    // The old helper passed `undefined` as the base during SSR, so every
    // relative URL threw and was rejected — server HTML then disagreed with
    // the client render after hydration.
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
