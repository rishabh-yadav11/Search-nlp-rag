/**
 * The dashboard must never render a chat conversation's text: the backend used to send the session title
 * (the first 60 characters of the user's own question) in each `/analytics/chat` top-N row and the page
 * printed it. It now sends the opaque session id, from which the page renders a short label.
 */
import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearMeCache } from '../../lib/auth'
import { formatEpochDateTime } from '../../lib/format'
import AnalyticsDashboardPage from './page'

type StubResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
  text: () => Promise<string>
}

function jsonResponse(data: unknown): StubResponse {
  return { ok: true, status: 200, json: async () => data, text: async () => JSON.stringify(data) }
}

const ADMIN = { id: 1, email: 'admin@example.com', name: 'Admin', role: 'admin', is_active: true }

const SUMMARY = {
  searches_total: 10,
  searches_today: 4,
  zero_result_rate: 0,
  weak_result_rate: 0,
  filtered_rate: 0,
  cache_hit_rate: 0,
  avg_latency_ms: 12,
  clicks_total: 0,
  top_queries: [],
  click_positions: {},
  click_top_queries: [],
}

// ts values chosen so the rendered "Updated" cells differ per row.
const COST_TS = 1_700_000_000
const TOKEN_TS = 1_700_000_500

const COST_ID = 'a1b2c3d4e5f60718'
const TOKEN_ID = '9988776655443322'

// Row shape: [sessionId, messages, value, updatedAt, ...] — plus a trailing `title` the page must
// never render.
const CHAT = {
  sessions: 2,
  users: 2,
  messages: 11,
  total_tokens: 12345,
  total_cost: 0.0234,
  avg_latency_ms: 900,
  sessions_today: 2,
  top_by_cost: [[COST_ID, 4, 0.0123, COST_TS, 'What were the Q3 revenue drivers?']],
  top_by_tokens: [[TOKEN_ID, 7, 12345, TOKEN_TS, 'Summarise this week in cybersecurity']],
}

const STALE_TITLE = 'What were the Q3 revenue drivers?'
const OTHER_TITLE = 'Summarise this week in cybersecurity'

function stubFetch(chat: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL): Promise<StubResponse> => {
      const url = String(input)
      if (url.includes('/api/auth/me')) return jsonResponse(ADMIN)
      if (url.includes('/analytics/chat')) return jsonResponse(chat)
      if (url.includes('/analytics/summary')) return jsonResponse(SUMMARY)
      throw new Error(`unexpected fetch: ${url}`)
    })
  )
}

async function renderDashboard(chat: unknown = CHAT) {
  stubFetch(chat)
  render(<AnalyticsDashboardPage />)
  await screen.findByText('Chat usage')
}

function cellsOf(label: string): string[] {
  const row = screen.getByText(label).closest('tr')
  if (!row) throw new Error(`no row rendered for ${label}`)
  return Array.from(row.querySelectorAll('td')).map((c) => c.textContent ?? '')
}

// The shared formatter, imported rather than re-derived. The assertion is about per-row identity, not
// the date format.
function renderedAt(ts: number): string {
  return formatEpochDateTime(ts)
}

beforeEach(() => {
  clearMeCache()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('AnalyticsDashboardPage — chat rows identify a session, not its text', () => {
  it('renders one labelled row per entry with its message count, value and updated time', async () => {
    await renderDashboard()

    const labels = screen.getAllByText(/^Session \w+$/).map((el) => el.textContent)
    expect(labels).toHaveLength(2)

    const costCells = cellsOf(`Session ${COST_ID.slice(0, 8)}`)
    expect(costCells[1]).toBe('4')
    expect(costCells[2]).toContain('0.0123')
    expect(costCells[3]).toBe(renderedAt(COST_TS))

    const tokenCells = cellsOf(`Session ${TOKEN_ID.slice(0, 8)}`)
    expect(tokenCells[1]).toBe('7')
    expect(tokenCells[2]).toBe('12,345')
    expect(tokenCells[3]).toBe(renderedAt(TOKEN_TS))
  })

  it('keeps the full session id reachable on the row for admins', async () => {
    await renderDashboard()

    expect(screen.getByText(`Session ${COST_ID.slice(0, 8)}`).getAttribute('title')).toBe(COST_ID)
  })

  it('never renders a title a stale or hostile backend may still send', async () => {
    await renderDashboard()

    expect(document.body.textContent).not.toContain(STALE_TITLE)
    expect(document.body.textContent).not.toContain(OTHER_TITLE)
    const firstCells = screen
      .getAllByText(/^Session \w+$/)
      .map((el) => el.textContent)
      .sort()
    expect(firstCells).toEqual([`Session ${COST_ID.slice(0, 8)}`, `Session ${TOKEN_ID.slice(0, 8)}`].sort())
  })
})

describe('AnalyticsDashboardPage — empty chat stats', () => {
  it('says so for both tables instead of rendering rows', async () => {
    await renderDashboard({ ...CHAT, top_by_cost: [], top_by_tokens: [] })

    expect(screen.getAllByText('No chat activity yet.')).toHaveLength(2)
    expect(screen.queryAllByText(/^Session \w+$/)).toHaveLength(0)
  })
})


// --- A failed feed is shown as failed, never as an all-zero report ---

const UNAVAILABLE_SUMMARY = { error: 'analytics unavailable', detail: 'the analytics store could not be read' }
const UNAVAILABLE_CHAT = { error: 'chat analytics unavailable', detail: 'the analytics store could not be read' }

function statusResponse(status: number, data: unknown): StubResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  }
}

function stubFeeds(summary: StubResponse, chat: StubResponse) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL): Promise<StubResponse> => {
      const url = String(input)
      if (url.includes('/api/auth/me')) return jsonResponse(ADMIN)
      if (url.includes('/analytics/chat')) return chat
      if (url.includes('/analytics/summary')) return summary
      throw new Error(`unexpected fetch: ${url}`)
    })
  )
}

async function renderFeeds(summary: StubResponse, chat: StubResponse) {
  stubFeeds(summary, chat)
  render(<AnalyticsDashboardPage />)
  await screen.findByText(/^Analytics unavailable/)
}

/** Whole card text (label + value + hint) for a metric card. */
function cardText(label: string): string {
  return screen.getByText(label).parentElement?.textContent ?? ''
}

describe('AnalyticsDashboardPage — a store that cannot be read is not a quiet day', () => {
  it('shows an unavailable state, not zero cards, when both feeds answer 503', async () => {
    await renderFeeds(statusResponse(503, UNAVAILABLE_SUMMARY), statusResponse(503, UNAVAILABLE_CHAT))

    expect(screen.getByText(/Analytics unavailable/)).toBeTruthy()
    expect(screen.getByText(/Search analytics \(unavailable\)/)).toBeTruthy()
    expect(screen.getByText(/Chat usage \(unavailable\)/)).toBeTruthy()
    // The zeroed report is the bug: none of its cards may render.
    for (const label of ['Searches today', 'Zero-result rate', 'Clicks', 'Chat users', 'Total tokens']) {
      expect(screen.queryByText(label)).toBeNull()
    }
    expect(screen.queryByText('No data yet.')).toBeNull()
    expect(screen.queryByText('No chat activity yet.')).toBeNull()
    expect(document.body.textContent).not.toMatch(/Updated \d/)
  })

  it('treats a 503 with no error key as a failure on the status line alone', async () => {
    // An intermediary can answer 503 with an HTML error page or empty body, so the status line has to
    // be load-bearing on its own, not only the `error` key.
    await renderFeeds(statusResponse(503, '<html>502 Bad Gateway</html>'), statusResponse(200, CHAT))

    expect(screen.getByText(/Search analytics \(unavailable\)/)).toBeTruthy()
    expect(screen.getByText(/HTTP 503/)).toBeTruthy()
    expect(screen.queryByText('Searches today')).toBeNull()
  })

  it('still detects the legacy 200-with-error body the backend used to send', async () => {
    // Status 200 plus an `error` key is what the backend sent during a Redis outage; treating it as data
    // produced the all-zero dashboard.
    await renderFeeds(statusResponse(200, UNAVAILABLE_SUMMARY), statusResponse(200, UNAVAILABLE_CHAT))

    expect(screen.getByText(/Search analytics \(unavailable\)/)).toBeTruthy()
    expect(screen.queryByText('Searches today')).toBeNull()
    expect(screen.queryByText('Chat users')).toBeNull()
  })

  it('keeps the healthy feed rendering when only the other one is down', async () => {
    await renderFeeds(statusResponse(503, UNAVAILABLE_SUMMARY), jsonResponse(CHAT))

    expect(screen.getByText(/Search analytics \(unavailable\)/)).toBeTruthy()
    expect(screen.queryByText('Searches today')).toBeNull()
    // The chat half is real data and must still be readable.
    await screen.findByText('Chat usage')
    expect(cardText('Chat users')).toContain('2')
    expect(screen.getAllByText(/^Session \w+$/)).toHaveLength(2)
  })

  it('renders the full report with no unavailable state when both feeds are healthy', async () => {
    stubFeeds(jsonResponse(SUMMARY), jsonResponse(CHAT))
    render(<AnalyticsDashboardPage />)
    await screen.findByText('Chat usage')

    expect(screen.queryByText(/unavailable/i)).toBeNull()
    expect(cardText('Searches today')).toContain('4')
    expect(cardText('Chat users')).toContain('2')
    expect(document.body.textContent).toMatch(/Updated \d/)
  })
})

describe('AnalyticsDashboardPage — a hung identity check is cancelled, not abandoned', () => {
  it('aborts the in-flight /api/auth/me when the 10 s race gives up', async () => {
    vi.useFakeTimers()
    // getMe logs the cancelled check; the assertion below is on the abort, not the log.
    vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const captured: { signal: AbortSignal | null } = { signal: null }
      vi.stubGlobal(
        'fetch',
        vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
          if (String(input).includes('/api/auth/me')) {
            captured.signal = init?.signal ?? null
            const { promise, reject } = Promise.withResolvers<StubResponse>()
            init?.signal?.addEventListener('abort', () => {
              const err = new Error('The operation was aborted')
              err.name = 'AbortError'
              reject(err)
            })
            return promise
          }
          return Promise.resolve(jsonResponse(ADMIN))
        })
      )

      render(<AnalyticsDashboardPage />)
      expect(captured.signal).toBeTruthy()
      expect(captured.signal?.aborted).toBe(false)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000)
      })

      // Giving up on the promise is not enough: the socket must close, or the request outlives the race.
      expect(captured.signal?.aborted).toBe(true)
      expect(screen.getByText('Analytics unavailable: identity check timed out')).toBeTruthy()
    } finally {
      vi.restoreAllMocks()
      vi.useRealTimers()
    }
  })
})
