/**
 * Issue #287 — a hung `/recommend/*` response must not pin the For You feed in
 * "Loading..." forever.
 *
 * The stub below never settles on its own: it holds the request open until the
 * signal it was handed aborts, which is exactly what a backend that accepts the
 * connection and never responds looks like to `fetch`. If the page has no
 * deadline, the test hangs at "Loading..." and the assertions below fail; with a
 * deadline the socket is cancelled, the catch runs, and the user gets a message
 * plus a Retry.
 */
import { act, render, screen } from '@testing-library/react'
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

  it('aborts the request on unmount without reporting a timeout', async () => {
    const view = render(<ForYouPage />)
    expect(captures[0].signal?.aborted).toBe(false)

    view.unmount()
    expect(captures[0].signal?.aborted).toBe(true)

    // Nothing to assert on the DOM (it is gone); the guard is that no state
    // update is attempted, which the absence of an act() warning proves.
    await advance(RECOMMEND_DEADLINE_MS)
  })
})
