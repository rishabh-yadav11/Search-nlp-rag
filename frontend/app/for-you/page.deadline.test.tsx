/**
 * The stub never settles on its own: it holds the request open until its signal aborts, which is what a
 * backend that accepts the connection and never responds looks like to `fetch`.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RECOMMEND_DEADLINE_MS } from '../lib/deadline'
import ForYouPage from './page'

type SignalCapture = { url: string; signal: AbortSignal | null }

let captures: SignalCapture[] = []

/** A signed-out answer for the shared top bar's identity check. */
const SIGNED_OUT = { ok: false, status: 401, json: async () => null } as unknown as Response

/**
 * Feed and beacon requests only. The shared top bar's identity check is answered immediately: a hung
 * `/api/auth/me` would be a second, unrelated thing to account for.
 */
function feedCaptures(): SignalCapture[] {
  return captures.filter((c) => !c.url.includes('/api/auth/me'))
}

function hangingFetch() {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url.includes('/api/auth/me')) return Promise.resolve(SIGNED_OUT)
    const signal = init?.signal ?? null
    captures.push({ url, signal })
    const { promise, reject } = Promise.withResolvers<Response>()
    if (signal) {
      signal.addEventListener('abort', () => {
        const err = new Error('The operation was aborted')
        err.name = 'AbortError'
        reject(err)
      })
    }
    return promise
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  captures = []
  vi.stubGlobal('fetch', hangingFetch())
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('ForYouPage — a hung feed request does not pin the loading state', () => {
  it('leaves loading and shows a retryable error once the deadline elapses', async () => {
    render(<ForYouPage />)
    expect(screen.getByText('Loading...')).toBeTruthy()

    await advance(RECOMMEND_DEADLINE_MS - 1)
    expect(screen.getByText('Loading...')).toBeTruthy()

    await advance(1)
    expect(screen.queryByText('Loading...')).toBeNull()
    expect(screen.getByText(/did not respond in time/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('aborts the in-flight request on timeout instead of leaving it open', async () => {
    render(<ForYouPage />)
    expect(feedCaptures()).toHaveLength(1)
    expect(feedCaptures()[0].signal?.aborted).toBe(false)

    await advance(RECOMMEND_DEADLINE_MS)

    expect(feedCaptures()[0].signal?.aborted).toBe(true)
  })

  it('refetches with a fresh deadline when Retry is pressed', async () => {
    render(<ForYouPage />)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(feedCaptures()).toHaveLength(1)

    await act(async () => {
      screen.getByRole('button', { name: 'Retry' }).click()
    })
    expect(feedCaptures()).toHaveLength(2)
    expect(screen.getByText('Loading...')).toBeTruthy()
    expect(feedCaptures()[1].signal?.aborted).toBe(false)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(feedCaptures()[1].signal?.aborted).toBe(true)
  })

  it('keeps a still-mounted feed out of the cancelled request’s error path', async () => {
    // Switching feed type cancels request #1 and starts #2 while the page stays mounted, so a stray
    // timeout from the abandoned request has a live component to render into. Unmounting cannot do
    // this: the DOM is gone, so the assertion passes either way. #1 hangs until the deadline and #2
    // answers, so exactly one request fails.
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.includes('/api/auth/me')) return Promise.resolve(SIGNED_OUT)
        captures.push({ url, signal: init?.signal ?? null })
        if (feedCaptures().length > 1) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              articles: [{ id: 9, title: 'Trending deal', url: 'https://vccircle.com/news/trending' }],
            }),
          } as unknown as Response)
        }
        const { promise, reject } = Promise.withResolvers<Response>()
        init?.signal?.addEventListener('abort', () => {
          const err = new Error('The operation was aborted')
          err.name = 'AbortError'
          reject(err)
        })
        return promise
      })
    )

    render(<ForYouPage />)
    await advance(0)
    expect(feedCaptures()).toHaveLength(1)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Trending' }))
    })
    expect(feedCaptures()).toHaveLength(2)

    // Past the deadline, so the abandoned request #1 fails for real.
    await advance(RECOMMEND_DEADLINE_MS)
    expect(feedCaptures()[0].signal?.aborted).toBe(true)

    // The feed the user is on moved on, so #1's timeout must not be reported...
    expect(screen.queryByText(/did not respond in time/)).toBeNull()
    // ...and #2's answer is what the page shows.
    expect(screen.getByText('Trending deal')).toBeTruthy()
  })
})

describe('ForYouPage — the click-tracking beacon is bounded too', () => {
  it('aborts a hung interaction POST instead of leaving the socket open', async () => {
    // Feed resolves so a card renders; the interaction POST then hangs.
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.includes('/api/auth/me')) return Promise.resolve(SIGNED_OUT)
        captures.push({ url, signal: init?.signal ?? null })
        if (url.includes('/recommend/interaction')) {
          const { promise, reject } = Promise.withResolvers<Response>()
          init?.signal?.addEventListener('abort', () => {
            const err = new Error('The operation was aborted')
            err.name = 'AbortError'
            reject(err)
          })
          return promise
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            articles: [{ id: 1, title: 'Deal 1', url: 'https://vccircle.com/news/deal-1' }],
          }),
        } as unknown as Response)
      })
    )

    render(<ForYouPage />)
    // Fake timers are installed, so `findBy*` would wait on a clock that is not moving.
    await advance(0)
    const link = screen.getByText('Deal 1')
    await act(async () => {
      fireEvent.click(link)
    })

    const beacon = captures.find((c) => c.url.includes('/recommend/interaction'))
    expect(beacon).toBeTruthy()
    expect(beacon?.signal?.aborted).toBe(false)

    await advance(RECOMMEND_DEADLINE_MS)
    expect(beacon?.signal?.aborted).toBe(true)
  })
})
