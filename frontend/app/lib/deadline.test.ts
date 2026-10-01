/**
 * The deadline helper that bounds every `fetch` call site. The behaviour under
 * test is the guarantee the call sites rely on: the signal fires at the
 * deadline, `clear()` cancels it, and an unmount abort stays distinguishable
 * from a timeout via `timedOut()`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createDeadline } from './deadline'

const MS = 15_000

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('createDeadline', () => {
  it('aborts the signal once the deadline elapses', () => {
    const d = createDeadline(MS)
    expect(d.signal.aborted).toBe(false)

    vi.advanceTimersByTime(MS - 1)
    expect(d.signal.aborted).toBe(false)

    vi.advanceTimersByTime(1)
    expect(d.signal.aborted).toBe(true)
    d.clear()
  })

  it('never fires after clear()', () => {
    const d = createDeadline(MS)
    d.clear()

    vi.advanceTimersByTime(MS * 10)
    expect(d.signal.aborted).toBe(false)
  })

  it('reports timedOut() only when the deadline itself fired', () => {
    const cleared = createDeadline(MS)
    cleared.clear()
    vi.advanceTimersByTime(MS)
    expect(cleared.timedOut()).toBe(false)

    const fired = createDeadline(MS)
    vi.advanceTimersByTime(MS)
    expect(fired.timedOut()).toBe(true)
    fired.clear()
  })

  it('aborts immediately when the base signal is already aborted', () => {
    const base = new AbortController()
    base.abort()

    const d = createDeadline(MS, base.signal)
    expect(d.signal.aborted).toBe(true)
    // An unmount abort must not be mistaken for a timeout, or the call site
    // would surface a spurious error to a component that is going away.
    expect(d.timedOut()).toBe(false)
    d.clear()
  })

  it('propagates a later base abort without marking it as a timeout', () => {
    const base = new AbortController()
    const d = createDeadline(MS, base.signal)

    base.abort()
    expect(d.signal.aborted).toBe(true)
    expect(d.timedOut()).toBe(false)
    d.clear()
  })

  it('stops listening to the base signal after clear()', () => {
    const base = new AbortController()
    const d = createDeadline(MS, base.signal)
    d.clear()

    base.abort()
    expect(d.signal.aborted).toBe(false)
  })
})
