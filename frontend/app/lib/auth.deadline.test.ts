/**
 * A hung auth service must be cancelled, not merely abandoned: `Promise.race` only drops the promise,
 * leaving the `/api/auth/me` fetch in flight (one leaked request per dashboard poll), so the deadline
 * aborts the socket. A timeout is a transport failure, not a logout, so `getMe` rethrows.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearMeCache, getMe } from './auth'
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
 * `getMe()` attaches its rejection handler only after `await fetch` resumes, so the rejection is
 * unhandled for a tick. Settling it into a value here keeps the harness quiet.
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

  clearMeCache()
  vi.stubGlobal('fetch', hangingFetch())
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()

  clearMeCache()
})

/** Advance the clock; `advanceTimersByTimeAsync` also flushes the microtasks the rejection schedules. */
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
    // A live signal alone would pass even with no deadline at all, so also pin that it stays pending.
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

  it('leaves the session alone on a timeout instead of treating it as a logout', async () => {
    // The credential is an httpOnly cookie JS cannot read or clear, so a timeout must not resolve
    // `null` (this function's "definitive logged out" sentinel) nor cache one — a cancelled read
    // that reads as a logout is a forced logout. It must reject, and the next call must hit the network.
    const pending = track(getMe())
    await advance(ME_DEADLINE_MS)
    expect((await pending).ok).toBe(false)

    track(getMe())
    expect(signals).toHaveLength(2)
  })

  it('is cancelled by the caller aborting its own signal, as the dashboard does', async () => {
    // The dashboard abandons `getMe()` at 10 s, which must cancel the socket rather than outlive the race.
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
    // Logging this as "failed to reach the auth service" would report a healthy auth service as down.
    expect(console.error).not.toHaveBeenCalled()
  })

  it('still logs a genuine transport failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))

    const pending = track(getMe(false, new AbortController().signal))
    expect((await pending).ok).toBe(false)
    expect(console.error).toHaveBeenCalledWith('getMe: failed to reach the auth service', expect.anything())
  })

  it('rejects, not resolves null, when a stalled body is cut off by the deadline', async () => {
    // undici's behaviour: headers resolve, then the body read rejects on abort. A never-settling stub
    // cannot model that and would leave getMe()'s result unobserved.
    vi.stubGlobal(
      'fetch',
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
        signals = [init?.signal ?? null]
        const { promise, reject } = Promise.withResolvers<never>()
        init?.signal?.addEventListener('abort', () => {
          const err = new Error('The operation was aborted')
          err.name = 'AbortError'
          reject(err)
        })
        // Headers arrive immediately; the body stalls until the abort lands.
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => promise,
        } as unknown as Response)
      })
    )

    const pending = track(getMe())
    await advance(ME_DEADLINE_MS)

    const result = await pending
    // Resolving `null` here would tell the dashboard the user is logged out when no body ever arrived.
    expect(result.ok).toBe(false)
    expect(signals[0]?.aborted).toBe(true)
    expect(console.error).not.toHaveBeenCalledWith(
      'getMe: failed to parse /api/auth/me response',
      expect.anything()
    )
  })

  it('still returns null (not a throw) for a genuinely malformed 200 body', async () => {
    // Callers may lack a .catch, so a real parse failure resolves null rather than throwing.
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.reject(new SyntaxError('Unexpected token <')),
        } as unknown as Response)
      )
    )

    const pending = track(getMe())
    await advance(0)
    const result = await pending
    expect(result.ok).toBe(true)
    if (result.ok) expect(result.value).toBeNull()
  })
})
