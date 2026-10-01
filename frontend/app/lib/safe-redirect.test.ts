import { describe, expect, it } from 'vitest'
import { isSafeRedirect } from './safe-url'

/**
 * The guard this replaced, copied from the two call sites before they were consolidated. Kept as a
 * reference oracle: the suite below asserts the consolidated guard is never MORE PERMISSIVE, so a
 * future "simplification" cannot quietly reopen an open redirect.
 */
function preConsolidationGuard(next: unknown): boolean {
  if (typeof next !== 'string' || next.length === 0) return false
  if (!next.startsWith('/')) return false
  if (next.startsWith('//')) return false
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(next)) return false
  return true
}

/** Every hostile or degenerate `?next=` value in one corpus. */
const HOSTILE = [
  // Protocol-relative escapes.
  '//evil.com',
  '//evil.com/path',
  '///evil.com',
  '\\\\evil.com',
  '/\\evil.com',
  '\\/evil.com',
  // The URL parser REMOVES tab/CR/LF, so these read as `//evil.com` in a browser while looking like an
  // ordinary local path. Both pre-consolidation copies let them through.
  '/\t/evil.com',
  '/\n/evil.com',
  '/\r/evil.com',
  '/\u0000/evil.com',
  '/\u0000//evil.com',
  '//evil.com\u0000',
  '/\u001f/evil.com',
  // Absolute references and schemes.
  'https://evil.com',
  'http://evil.com',
  'javascript:alert(1)',
  'data:text/html,<script>alert(1)</script>',
  'HTTPS://evil.com',
  'file:///etc/passwd',
  'vbscript:msgbox(1)',
  // Not a root-relative path at all.
  'evil.com',
  'chat',
  '',
  '   ',
  ' /chat',
  null,
  undefined,
  42,
  {},
  ['/chat'],
  Symbol('next'),
]

const SAFE = [
  '/chat',
  '/',
  '/analytics/dashboard',
  '/search?q=hello&page=2',
  '/articles/some-slug#section',
  '/path/with%20encoded%20space',
  '/trailing/slash/',
  '/unicode/日本語',
  // A same-origin PATH whose first segment is the literal text "javascript:"; the browser and Next's
  // router both resolve it against the current origin, so it is not a script URL.
  '/javascript:alert(1)',
]

describe('isSafeRedirect — rejects every hostile next value', () => {
  it.each(HOSTILE)('rejects %j', (value) => {
    expect(isSafeRedirect(value)).toBe(false)
  })
})

describe('isSafeRedirect — accepts root-relative paths', () => {
  it.each(SAFE)('accepts %j', (value) => {
    expect(isSafeRedirect(value)).toBe(true)
  })
})

describe('isSafeRedirect — consolidation is not a weakening', () => {
  // accepted_new ⊆ accepted_old for every input, over a corpus hostile, degenerate and legitimate in
  // equal measure.
  const CORPUS = [...HOSTILE, ...SAFE]

  it.each(CORPUS)('is no more permissive than the pre-merge copy for %j', (value) => {
    if (isSafeRedirect(value)) {
      expect(preConsolidationGuard(value)).toBe(true)
    }
  })

  it('closes the escapes both old copies missed', () => {
    // Inputs the pre-consolidation guard ACCEPTED and the browser resolves off-origin. `\\evil.com` is
    // not among them: it never starts with `/`, so the old guard already refused it.
    for (const escape of ['/\\evil.com', '/\t/evil.com', '/\n/evil.com', '/\u0000/evil.com']) {
      expect(preConsolidationGuard(escape)).toBe(true)
      expect(isSafeRedirect(escape)).toBe(false)
    }
  })
})

describe('isSafeRedirect — narrows, so callers cannot use an unchecked string', () => {
  it('is a type guard: true means the value is a usable string', () => {
    const raw: string | null = '/chat'
    if (isSafeRedirect(raw)) {
      // Compiles only because the guard narrows to `string`; the pre-merge signup copy returned `boolean`.
      const narrowed: string = raw
      expect(narrowed).toBe('/chat')
    } else {
      throw new Error('expected /chat to be safe')
    }
  })
})
