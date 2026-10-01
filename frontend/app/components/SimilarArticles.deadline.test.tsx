/**
 * The stub holds the request open until its signal aborts, which is what a silent backend looks like to
 * `fetch`. The deadline lives in `app/lib/similar.ts` on the shared batch request, so these tests drive
 * the card but assert on the signal that module handed to `fetch`.
 *
 * The component is imported fresh per test because `similar.ts` keeps its cache and in-flight map at
 * module scope: without a reset, a card from an earlier test is served from memory and issues no request.
 */
import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RECOMMEND_DEADLINE_MS } from '../lib/deadline'
import type SimilarArticlesComponent from './SimilarArticles'

let signals: (AbortSignal | null)[] = []
let SimilarArticles: typeof SimilarArticlesComponent

function hangingFetch() {
  return vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
    const signal = init?.signal ?? null
    signals.push(signal)
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

beforeEach(async () => {
  vi.useFakeTimers()
  signals = []
  vi.stubGlobal('fetch', hangingFetch())
  vi.resetModules()
  ;({ default: SimilarArticles } = await import('./SimilarArticles'))
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

describe('SimilarArticles — a hung request does not pin the loading state', () => {
  it('replaces the compact loading text with a retryable error at the deadline', async () => {
    render(<SimilarArticles articleId={1} compact />)
    expect(screen.getByText('Loading...')).toBeTruthy()

    await advance(RECOMMEND_DEADLINE_MS - 1)
    expect(screen.getByText('Loading...')).toBeTruthy()

    await advance(1)
    expect(screen.queryByText('Loading...')).toBeNull()
    expect(screen.getByText(/did not load in time/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('aborts the in-flight request on timeout instead of leaving it open', async () => {
    render(<SimilarArticles articleId={1} compact />)
    // The batch flush is a macrotask, so the request only exists after a tick.
    await advance(0)
    expect(signals[0]?.aborted).toBe(false)

    await advance(RECOMMEND_DEADLINE_MS)

    expect(signals[0]?.aborted).toBe(true)
  })

  it('shows the same error and retry in the card layout', async () => {
    render(<SimilarArticles articleId={1} />)
    await advance(RECOMMEND_DEADLINE_MS)

    expect(screen.getByText(/did not load in time/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('refetches with a fresh deadline when Retry is pressed', async () => {
    render(<SimilarArticles articleId={1} compact />)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(signals).toHaveLength(1)

    await act(async () => {
      screen.getByRole('button', { name: 'Retry' }).click()
    })
    // The batch flush is a macrotask (see `similar.ts`), so fake timers must move for request #2.
    await advance(0)
    expect(signals).toHaveLength(2)
    expect(signals[1]?.aborted).toBe(false)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(signals[1]?.aborted).toBe(true)
  })
})

describe('SimilarArticles — a cancelled card is silent while a live one is not', () => {
  // Unmounting destroys the DOM, so an assertion after `unmount()` sees a torn-down tree and passes
  // either way. Keeping the card MOUNTED and swapping its articleId abandons the old request and starts
  // a new one, giving a stray timeout from the abandoned request a live component to render into.
  it('renders no timeout message for a request the card stopped waiting on', async () => {
    // #1 hangs until the deadline and #2 answers: exactly one request fails, so a painted #1 failure
    // would be on screen.
    vi.stubGlobal(
      'fetch',
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
        const signal = init?.signal ?? null
        signals.push(signal)
        if (signals.length > 1) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
              results: [
                {
                  article_id: 2,
                  similar_articles: [
                    { id: 20, title: 'Answered', url: 'https://vccircle.com/news/answered' },
                  ],
                },
              ],
            }),
          } as unknown as Response)
        }
        const { promise, reject } = Promise.withResolvers<Response>()
        signal?.addEventListener('abort', () => {
          const err = new Error('The operation was aborted')
          err.name = 'AbortError'
          reject(err)
        })
        return promise
      })
    )

    const view = render(<SimilarArticles articleId={1} compact />)
    await advance(0)
    expect(signals).toHaveLength(1)

    view.rerender(<SimilarArticles articleId={2} compact />)
    expect(screen.getByText('Loading...')).toBeTruthy()
    // As above: the second id joins the next batch flush, a macrotask.
    await advance(0)
    expect(signals).toHaveLength(2)

    // Past the deadline, so the abandoned request #1 fails for real.
    await advance(RECOMMEND_DEADLINE_MS)
    expect(signals[0]?.aborted).toBe(true)

    // The card moved on, so #1's timeout must not be reported to the user...
    expect(screen.queryByText(/did not load in time/)).toBeNull()
    // ...and #2's answer must be what it actually shows.
    expect(screen.getByText('Answered')).toBeTruthy()
  })

  it('still reports the timeout on the card that is actually waiting', async () => {
    // The control for the test above: a single mounted card DOES surface the same elapsed time, so the
    // silence there comes from the abandoned request, not an inert assertion.
    const view = render(<SimilarArticles articleId={1} compact />)
    await advance(0)
    expect(signals).toHaveLength(1)

    await advance(RECOMMEND_DEADLINE_MS)
    expect(screen.getByText(/did not load in time/)).toBeTruthy()

    // And the card recovers on Retry rather than staying broken.
    await act(async () => {
      screen.getByRole('button', { name: 'Retry' }).click()
    })
    expect(screen.queryByText(/did not load in time/)).toBeNull()
  })
})
