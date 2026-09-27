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

/** Advance the clock, then flush the microtasks the rejection schedules. */
async function advance(ms: number) {
  await vi.advanceTimersByTimeAsync(ms)
}

describe('getMe — a hung auth service is cancelled, not leaked', () => {
  it('passes a deadline signal to fetch and holds the request open before it', async () => {
    let settled = false
    const pending = track(getMe()).then((r) => {
      settled = true
      return r
    })
    expect(signals).toHaveLength(1)
    expect(signals[0]).toBeTruthy()
    expect(signals[0]?.aborted).toBe(false)

    await advance(1)
    // Both halves matter: a live, un-aborted signal AND a promise that is
    // genuinely still pending. Asserting only the signal would pass even if
    // `getMe` were governed by no deadline at all.
    expect(signals[0]?.aborted).toBe(false)
    expect(settled).toBe(false)
    void pending
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

  it('is cancelled by the caller aborting its own signal, as the dashboard does', async () => {
    // The dashboard gives up on `getMe()` at 10 s and aborts its load
    // controller. That must cancel the socket, so nothing outlives the race.
    const caller = new AbortController()
    const pending = track(getMe(false, caller.signal))
    const signal = signals[0]

    await advance(9_000)
    expect(signal?.aborted).toBe(false)

    caller.abort()
    expect(signal?.aborted).toBe(true)
    expect((await pending).ok).toBe(false)
  })

  it('does not log a caller abort as a backend failure', async () => {
    const caller = new AbortController()
    const pending = track(getMe(false, caller.signal))

    caller.abort()
    expect((await pending).ok).toBe(false)
    // A cancelled load is a cancellation, not a fault. Logging it as
    // "failed to reach the auth service" would report a healthy auth service
    // as down on every dashboard unmount and every 10 s race timeout.
    expect(console.error).not.toHaveBeenCalled()
  })

  it('still logs a genuine transport failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))

    const pending = track(getMe(false, new AbortController().signal))
    expect((await pending).ok).toBe(false)
    expect(console.error).toHaveBeenCalledWith('getMe: failed to reach the auth service', expect.anything())
  })

  it('aborts a response whose headers arrived but whose body never completes', async () => {
    // The deadline must stay armed across the body read, not just the header
    // read: otherwise this leaks the socket and hangs getMe() forever.
    const bodySignal = new AbortController()
    vi.stubGlobal(
      'fetch',
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
        signals = [init?.signal ?? null]
        init?.signal?.addEventListener('abort', () => bodySignal.abort())
        return Promise.resolve({
          ok: true,
          status: 200,
          // A body stream that never produces and never errors.
          json: () => new Promise<never>(() => {}),
        } as unknown as Response)
      })
    )

    const pending = track(getMe())
    let settled = false
    void pending.then(() => {
      settled = true
    })

    await advance(ME_DEADLINE_MS - 1)
    expect(settled).toBe(false)

    await advance(1)
    expect(signals[0]?.aborted).toBe(true)
    expect(bodySignal.signal.aborted).toBe(true)
  })
})
