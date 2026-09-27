/**
 * Issue #287 — `getMe()` had no signal and no deadline. The analytics
 * dashboard polls every 30 s and races `getMe()` against a 10 s timeout, but a
 * `Promise.race` only abandons a promise: the underlying `/api/auth/me` fetch
 * stayed in flight, so a hung auth service leaked one request per tick. The
 * deadline makes that abandonment real by aborting the socket.
 *
 * `getMe` also must NOT treat a timeout as "logged out" — it rethrows so callers
 * can tell a transport failure from a definitive 401.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearMeCache, getMe, TOKEN_KEY } from './auth'
import { ME_DEADLINE_MS } from './deadline'

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

/**
 * `getMe()` attaches its rejection handler only after `await fetch` resumes, so
 * the rejection is unhandled for a tick. Settling it into a value here keeps
 * the harness quiet and still lets the test assert the failure.
 */
function track(promise: Promise<unknown>) {
  return promise.then(
    (value) => ({ ok: true, value }) as const,
    (err: unknown) => ({ ok: false, err }) as const
  )
}
beforeEach(() => {
  vi.useFakeTimers()
  signals = []
  localStorage.setItem(TOKEN_KEY, 'test-token')
  clearMeCache()
  vi.stubGlobal('fetch', hangingFetch())
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  localStorage.removeItem(TOKEN_KEY)
  clearMeCache()
})

/** Resolves 'pending' if `p` has not settled by the time this returns. */
function settledOrPending<T>(p: Promise<T>) {
  return Promise.race([p.then(() => 'settled' as const), Promise.resolve('pending' as const)])
}
/** Advance the clock, then flush the microtasks the rejection schedules. */
async function advance(ms: number) {
  await vi.advanceTimersByTimeAsync(ms)
}

describe('getMe — a hung auth service is cancelled, not leaked', () => {
  it('passes a deadline signal to fetch and holds the request open before it', async () => {
    const pending = track(getMe())
    expect(signals).toHaveLength(1)
    expect(signals[0]).toBeTruthy()
    expect(signals[0]?.aborted).toBe(false)

    await advance(1)
    expect(signals[0]?.aborted).toBe(false)
    expect(await settledOrPending(pending)).toBe('pending')
  })

  it('aborts the in-flight request at the deadline', async () => {
    const pending = track(getMe())
    const signal = signals[0]

    await advance(ME_DEADLINE_MS - 1)
    expect(signal?.aborted).toBe(false)

    await advance(1)
    expect(signal?.aborted).toBe(true)
    expect((await pending).ok).toBe(false)
  })

  it('preserves the auth token on timeout instead of treating it as a logout', async () => {
    const pending = track(getMe())
    await advance(ME_DEADLINE_MS)
    expect((await pending).ok).toBe(false)

    expect(localStorage.getItem(TOKEN_KEY)).toBe('test-token')
  })

  it('still cancels the request when the dashboard abandons its race at 10 s', async () => {
    // The dashboard gives up on `getMe()` at 10 s. The request must still die
    // before the next 30 s poll tick, rather than surviving as a leaked fetch.
    const pending = track(getMe())
    const signal = signals[0]

    await advance(10_000)
    expect(signal?.aborted).toBe(false)

    await advance(ME_DEADLINE_MS - 10_000)
    expect(signal?.aborted).toBe(true)
    expect((await pending).ok).toBe(false)
  })
})
