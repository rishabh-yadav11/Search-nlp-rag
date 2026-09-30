/**
 * The tag filter reaches the backend through the SAME path as every other
 * facet: `Filters` state keys are spread verbatim into the `/search` query
 * string, so the state key and the backend's `tag` parameter are the same
 * contract. A mismatch is invisible until a request is issued, so these tests
 * assert the URL the page actually requested rather than any internal state.
 *
 * The suggestions come from a `/facets` response that may predate the filter
 * (no `tags` key) — an older backend must degrade to an empty datalist with
 * the input still usable as free text, since a tag filter is never inferred
 * from the query.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SearchPage from './page'

type StubResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
  text: () => Promise<string>
}

function jsonResponse(data: unknown, status = 200): StubResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  }
}

const SEARCH_RESULT = { query: 'fintech', results: [], cached: false, latency_ms: 1 }

let facetPayload: unknown
let searchUrls: string[]

function stubFetch() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL): Promise<StubResponse> => {
      const url = String(input)
      if (url.includes('/facets')) return jsonResponse(facetPayload)
      if (url.includes('/search')) {
        searchUrls.push(url)
        return jsonResponse(SEARCH_RESULT)
      }
      return jsonResponse({ detail: 'not found' }, 404)
    })
  )
}

beforeEach(() => {
  searchUrls = []
  facetPayload = { industry: ['fintech'], dealtype: ['venture debt'], tags: ['IPO', 'VCC Startups'] }
  stubFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function renderWithFilters() {
  const view = render(<SearchPage />)
  // Let the mount-time /facets request settle before opening the panel.
  await act(async () => {})
  fireEvent.click(screen.getByRole('button', { name: 'Show filters' }))
  return view
}


async function submitQuery() {
  fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'fintech' } })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Search' }))
  })
  return searchUrls[searchUrls.length - 1]
}


describe('SearchPage — the tag filter', () => {
  it('sends the typed tag as the `tag` parameter, trimmed', async () => {
    await renderWithFilters()
    fireEvent.change(screen.getByLabelText('Tag filter'), { target: { value: '  IPO  ' } })

    const params = new URL(await submitQuery(), 'http://test.local').searchParams

    expect(params.get('q')).toBe('fintech')
    expect(params.get('top_k')).toBe('8')
    expect(params.get('tag')).toBe('IPO')
  })

  it('omits an all-whitespace tag rather than sending it blank', async () => {
    await renderWithFilters()
    fireEvent.change(screen.getByLabelText('Tag filter'), { target: { value: '   ' } })

    const url = await submitQuery()

    expect(new URL(url, 'http://test.local').searchParams.has('tag')).toBe(false)
    expect(url).not.toContain('tag=')
  })

  it('offers the /facets tags as suggestions', async () => {
    await renderWithFilters()

    const options = Array.from(
      document.querySelectorAll('#tag-options option'),
      (o) => (o as HTMLOptionElement).value
    )

    expect(options).toEqual(['IPO', 'VCC Startups'])
  })

  it('says the suggestions are only the popular subset', async () => {
    await renderWithFilters()

    // A silent 200-value cap reads as "these are all the tags".
    expect(screen.getByText(/top 200 by use/).textContent).toContain('tag')
  })

  it('stays a usable free-text field when /facets sends no tags', async () => {
    // A backend that has not shipped the tag facet yet.
    facetPayload = { industry: ['fintech'], dealtype: ['venture debt'] }

    await renderWithFilters()

    expect(document.querySelectorAll('#tag-options option')).toHaveLength(0)
    expect(screen.queryByRole('alert')).toBeNull()

    fireEvent.change(screen.getByLabelText('Tag filter'), { target: { value: 'Waaree Energies' } })
    const url = await submitQuery()
    expect(new URL(url, 'http://test.local').searchParams.get('tag')).toBe('Waaree Energies')
  })

  it('is cleared by the Clear filters button', async () => {
    await renderWithFilters()
    fireEvent.change(screen.getByLabelText('Tag filter'), { target: { value: 'IPO' } })

    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))

    const input = screen.getByLabelText('Tag filter') as HTMLInputElement
    expect(input.value).toBe('')
  })
})
