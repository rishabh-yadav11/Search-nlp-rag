/**
 * Request deadlines for the fetch call sites governed here: the For You feed,
 * the shared similar-articles batch, chat's JSON `api()` and `getMe()`, plus
 * the fire-and-forget beacons in those files that would otherwise hold a
 * socket open with nothing to release it.
 *
 * A backend that accepts the connection but never answers leaves a promise
 * pending forever: no rejection, so downstream `catch`/`finally` never runs
 * and a `loading` flag stays true — a spinner with no error and no way out.
 * A deadline that aborts the socket, rather than merely abandoning the
 * promise, is what turns that hang into a failure the UI can report.
 *
 * Where the bound is armed depends on who owns the request. A page owning its
 * own socket arms it at the call site and passes its unmount controller as
 * `base`, so unmount and deadline stay independently observable (call sites
 * gate state updates on `base.aborted`). A request SHARED across a view is
 * bounded inside its owning module with no `base` at all: one card unmounting
 * must not take the rest of the view's request down, so a timeout is the only
 * thing that may abort it.
 *
 * Not `AbortSignal.timeout`: it cannot be disarmed and cannot be told apart
 * from a caller's own abort. This helper needs `clear()` on settle and
 * `timedOut()` to report a timeout without mislabelling an unmount.
 */

/** `/recommend/*` is the one backend path that degrades silently. */
export const RECOMMEND_DEADLINE_MS = 15_000
/** Chat session CRUD and the non-stream send: a full LLM round trip. */
export const CHAT_API_DEADLINE_MS = 30_000
/**
 * `/api/auth/me`, polled every 30 s by the analytics dashboard. Must stay ABOVE
 * the dashboard's own 10 s identity race: it reports a transport failure by
 * rethrowing and ignores an `AbortError` by design, so a deadline firing first
 * would win that race with exactly that error and the user would get no
 * message at all.
 */
export const ME_DEADLINE_MS = 15_000
/**
 * The sign-out POST: fire-and-forget behind an immediate redirect, so nobody
 * waits on the answer — the bound only exists to release the socket.
 */
export const LOGOUT_DEADLINE_MS = 10_000

export interface Deadline {
  /** Pass to `fetch({ signal })`. Aborts on the deadline or on `base`. */
  signal: AbortSignal
  /** True only if the DEADLINE fired; a caller abort (unmount) leaves it false. */
  timedOut(): boolean
  /** Stop the timer and detach the `base` listener. Always call on settle. */
  clear(): void
}

/** Fires after `ms`, composed with an optional caller signal (`base`). A `base` abort propagates immediately and does not set `timedOut()`. */
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

/**
 * A request that exceeded its deadline. Distinct from a transport
 * `AbortError`, so call sites can tell a timeout from a cancellation.
 */
export class RequestTimeoutError extends Error {
  readonly ms: number

  constructor(ms: number) {
    super(`The request timed out after ${Math.round(ms / 1000)}s. Check your connection and try again.`)
    this.name = 'RequestTimeoutError'
    this.ms = ms
  }
}
