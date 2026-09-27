/**
 * Issue #287 — a hung `/recommend/similar/*` response must not pin a
 * SimilarArticles card in "Loading..." forever. The stub holds the request open
 * until its signal aborts, which is what a silent backend looks like to
 * `fetch`: without a deadline the card is stuck on the loading branch forever.
 */
import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RECOMMEND_DEADLINE_MS } from '../lib/deadline'
import SimilarArticles from './SimilarArticles'

let signals: (AbortSignal | null)[] = []

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

beforeEach(() => {
  vi.useFakeTimers()
  signals = []
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
    expect(signals).toHaveLength(2)
    expect(signals[1]?.aborted).toBe(false)
    await advance(RECOMMEND_DEADLINE_MS)
    expect(signals[1]?.aborted).toBe(true)
  })

  it('aborts the request on unmount without reporting a timeout', async () => {
    const view = render(<SimilarArticles articleId={1} compact />)
    view.unmount()
    expect(signals[0]?.aborted).toBe(true)

    await advance(RECOMMEND_DEADLINE_MS)
  })
})
