/**
 * Aborts the socket, not just the promise, so a backend that accepts a connection and
 * never answers fails instead of hanging. Not `AbortSignal.timeout`: it cannot be
 * disarmed after a request settles, nor told apart from a caller's own abort.
 */

/** `/recommend/*` is the one backend path that degrades silently. */
export const RECOMMEND_DEADLINE_MS = 15_000
/** Chat session CRUD and the non-stream send: a full LLM round trip. */
export const CHAT_API_DEADLINE_MS = 30_000
/**
 * Must stay above the dashboard's own 10 s identity race: firing first would win it
 * with an `AbortError` the dashboard ignores by design, and the user would see nothing.
 */
export const ME_DEADLINE_MS = 15_000
/** Fire-and-forget behind an immediate redirect, so the bound exists only to release the socket. */
export const LOGOUT_DEADLINE_MS = 10_000

export interface Deadline {
  signal: AbortSignal
  /** True only when the deadline fired — a caller's own abort stays silent, not a timeout. */
  timedOut(): boolean
  /** Must run when the request settles, or the timer and `base` listener outlive it. */
  clear(): void
}

/**
 * A request SHARED across a view passes no `base`, so one card unmounting cannot abort
 * the rest of the view's request; a `base` abort propagates without setting `timedOut()`.
 */
export function createDeadline(ms: number, base?: AbortSignal | null): Deadline {
  const controller = new AbortController()
  let fired = false

  const onBaseAbort = () => controller.abort()
  if (base) {
    if (base.aborted) controller.abort()
    else base.addEventListener('abort', onBaseAbort)
  }

  const timer = setTimeout(() => {
    fired = true
    controller.abort()
  }, ms)

  let cleared = false
  return {
    signal: controller.signal,
    timedOut: () => fired,
    clear() {
      if (cleared) return
      cleared = true
      clearTimeout(timer)
      base?.removeEventListener('abort', onBaseAbort)
    },
  }
}

/** An elapsed deadline, kept distinct from a transport `AbortError` so callers report "too slow", not "cancelled". */
export class RequestTimeoutError extends Error {
  readonly ms: number

  constructor(ms: number) {
    super(`The request timed out after ${Math.round(ms / 1000)}s. Check your connection and try again.`)
    this.name = 'RequestTimeoutError'
    this.ms = ms
  }
}
