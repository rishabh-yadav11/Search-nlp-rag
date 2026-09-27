/**
 * Issue #325 — the chat `SourceList` decided `href` safety with its own inline
 * `/^https?:\/\//` test instead of the shared `isSafeUrl` hardened in #246.
 *
 * These assertions are on the rendered DOM of the real page, not on the guard:
 * what matters is whether a backend-supplied source URL ever reaches an `a`.
 * Each unsafe payload is rendered as a whole thread so the verdict is observed
 * exactly where a reader would click it.
 */
import { fireEvent, render, screen } from '@testing-library/react'
import type * as DataVizModule from './DataViz'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { TOKEN_KEY } from '../lib/auth'
import ChatPage from './page'

// `SourceList` renders `SimilarArticles` per source, which fetches on mount.
vi.mock('../components/SimilarArticles', () => ({
  default: ({ articleId }: { articleId: number }) => <span data-testid={`similar-${articleId}`} />,
}))

// `AnswerBody` renders chart blocks through DataViz; nothing here is about them.
vi.mock('./DataViz', async () => {
  const actual = await vi.importActual<typeof DataVizModule>('./DataViz')
  return {
    ...actual,
    default: () => <span data-testid="data-viz" />,
  }
})

const SESSION = { id: 's1', title: 'Budget', created_at: 1_700_000_000, updated_at: 1_700_000_100 }

type Payload = { id: number; title: string; url: string; score: number }

/**
 * The escape spellings #246 was written for, in the two forms that matter:
 * bare (`//evil.com`, `/\evil.com`, …) and carrying a scheme
 * (`https:/\evil.com`, …). The browser resolves every one of them to
 * `https://evil.com/`, so each must end up inert rather than clickable.
 */
const UNSAFE: Payload[] = [
  { id: 1, title: 'Script scheme', url: "javascript:fetch(localStorage.getItem('vccircle_auth_token'))", score: 1 },
  { id: 2, title: 'Data scheme', url: 'data:text/html,<script>alert(1)</script>', score: 1 },
  { id: 3, title: 'Bare protocol relative', url: '//evil.com', score: 1 },
  { id: 4, title: 'Bare double backslash', url: '\\\\evil.com', score: 1 },
  { id: 5, title: 'Bare slash backslash', url: '/\\evil.com', score: 1 },
  { id: 6, title: 'Bare backslash slash', url: '\\/evil.com', score: 1 },
  { id: 7, title: 'Scheme slash backslash', url: 'https:/\\evil.com', score: 1 },
  { id: 8, title: 'Scheme double backslash', url: 'https:\\\\evil.com', score: 1 },
  { id: 9, title: 'Scheme backslash slash', url: 'https:\\/evil.com', score: 1 },
  { id: 10, title: 'NUL control escape', url: 'https:\x00//evil.com', score: 1 },
  { id: 11, title: 'Leading space script scheme', url: ' javascript:alert(1)', score: 1 },
]

const SAFE: Payload[] = [
  { id: 21, title: 'Absolute article', url: 'https://vccircle.com/news/deal-123', score: 1 },
  { id: 22, title: 'Same-origin relative', url: '/articles/9', score: 1 },
]

const SOURCES = [...UNSAFE, ...SAFE]

const MESSAGES = [
  { id: 1, role: 'user', content: 'question', created_at: 1_700_000_000 },
  { id: 2, role: 'assistant', content: 'ANSWER', created_at: 1_700_000_010, sources: SOURCES },
]

type StubResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
  text: () => Promise<string>
}

function jsonResponse(data: unknown): StubResponse {
  return { ok: true, status: 200, json: async () => data, text: async () => JSON.stringify(data) }
}

beforeEach(() => {
  localStorage.setItem(TOKEN_KEY, 'test-token')
  Element.prototype.scrollTo = function scrollTo() {}
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL): Promise<StubResponse> => {
      const url = String(input)
      if (url.endsWith('/api/chat/sessions')) return jsonResponse([SESSION])
      if (url.includes('/api/chat/sessions/')) return jsonResponse({ ...SESSION, messages: MESSAGES })
      return jsonResponse({ similar_articles: [] })
    })
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

/** Opens the thread and expands the sources list. */
async function renderSources() {
  render(<ChatPage />)
  fireEvent.click(await screen.findByLabelText('Open conversation: Budget'))
  await screen.findByText('ANSWER')
  fireEvent.click(screen.getByRole('button', { name: /Show sources/ }))
  await screen.findByText('Script scheme')
  return screen.getByRole('list')
}

const hrefsIn = (list: HTMLElement) => Array.from(list.querySelectorAll('a')).map((a) => a.getAttribute('href'))

describe('ChatPage source list — an unsafe source URL is never clickable', () => {
  it.each(UNSAFE.map((s) => [s.title, s.url] as const))(
    'renders %s inert rather than as a link',
    async (_title, url) => {
      const list = await renderSources()
      expect(hrefsIn(list)).not.toContain(url)
      // The title still renders, so the source is visible but not clickable.
      expect(screen.getByText(_title)).toBeTruthy()
    }
  )

  it('links only the two safe sources', async () => {
    const list = await renderSources()
    expect(hrefsIn(list)).toEqual(['https://vccircle.com/news/deal-123', '/articles/9'])
  })

  it('carries no payload anywhere in an href', async () => {
    const list = await renderSources()
    for (const { url } of UNSAFE) {
      expect(hrefsIn(list).some((href) => href?.includes('evil.com'))).toBe(false)
      expect(hrefsIn(list).some((href) => href?.startsWith('javascript:'))).toBe(false)
      expect(hrefsIn(list).some((href) => href?.startsWith('data:'))).toBe(false)
    }
  })
})
