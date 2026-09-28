/**
 * Issue #287 — a hung `/recommend/*` response must not pin the For You feed in
 * "Loading..." forever.
 *
 * The stub below never settles on its own: it holds the request open until the
 * signal it was handed aborts, which is exactly what a backend that accepts the
 * connection and never responds looks like to `fetch`. If the page has no
 * deadline the socket is cancelled, the catch runs, and the user gets a message
 * plus a Retry.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RECOMMEND_DEADLINE_MS } from '../lib/deadline'
import ForYouPage from './page'

type SignalCapture = { url: string; signal: AbortSignal | null }

let captures: SignalCapture[] = []
function hangingFetch() {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const signal = init?.signal ?? null
    captures.push({ url: String(input), signal })
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
    expect(captures).toHaveLength(1)
    expect(captures[0].signal?.aborted).toBe(false)

    await advance(RECOMMEND_DEADLINE_MS)

    expect(captures[0].signal?.aborted).toBe(true)
  })

  it('refetches with a fresh deadline when Retry is pressed', async () => {
    render(<ForYouPage />)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(captures).toHaveLength(1)

    await act(async () => {
      screen.getByRole('button', { name: 'Retry' }).click()
    })
    expect(captures).toHaveLength(2)
    expect(screen.getByText('Loading...')).toBeTruthy()
    // The retry must be able to reach the same deadline again.
    expect(captures[1].signal?.aborted).toBe(false)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(captures[1].signal?.aborted).toBe(true)
  })

  it('keeps a still-mounted feed out of the cancelled request’s error path', async () => {
    // The observable form of "unmounting does not report a timeout". Unmounting
    // destroys the DOM, so an assertion afterwards can only ever see a
    // torn-down tree — it passes whether or not the page misbehaved, which is
    // why the old unmount test here asserted nothing. The feed type is switched
    // instead, which cancels request #1 and starts request #2 while the page
    // stays mounted: a stray timeout message from the abandoned request has a
    // live component to render into and cannot hide.
    //
    // #1 hangs until the deadline and #2 answers, so exactly one request
    // fails: if the page painted the abandoned one, the text would show.
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        captures.push({ url, signal: init?.signal ?? null })
        if (captures.length > 1) {
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
    expect(captures).toHaveLength(1)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Trending' }))
    })
    expect(captures).toHaveLength(2)

    // Past the deadline, so the abandoned request #1 fails for real.
    await advance(RECOMMEND_DEADLINE_MS)
    expect(captures[0].signal?.aborted).toBe(true)

    // The feed the user is actually on moved on, so #1's timeout must not be
    // reported...
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
    // Fake timers are installed, so `findBy*` would wait on a clock that is
    // not moving. Flush the feed's microtasks explicitly instead.
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
