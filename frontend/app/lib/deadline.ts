/**
 * Request deadlines for the `fetch` call sites this helper now governs: the
 * For You feed, the shared similar-articles batch, chat's JSON `api()`
 * helper and `getMe()` — plus the two fire-and-forget beacons in those same
 * files (click tracking, sign-out) that would otherwise hold a socket open
 * with nothing to release it.
 *
 * A backend that accepts the connection but never answers leaves a promise
 * pending forever: no rejection, so every `catch`/`finally` downstream is never
 * reached and any `loading` flag stays true — the user gets a spinner with no
 * error and no way out. A deadline that aborts the underlying socket (rather
 * than merely abandoning the promise) is what turns that hang into a failure
 * the UI can report.
 *
 * This is a per-file budget set, not a claim that every fetch in the app is
 * routed through here. The search page, login, signup and the chat SSE stream
 * are untouched by this change and keep whatever bounds they already had, and
 * the search page's own fire-and-forget beacons keep the unbounded behaviour
 * they always had.
 *
 * Where the bound is armed depends on who owns the request. A page that owns
 * its own socket creates the deadline at the call site and passes its unmount
 * controller in as `base`, so unmount and deadline stay independently
 * observable (call sites gate their state updates on `base.aborted`, not on
 * the composed signal). A request SHARED across a view — the similar-articles
 * batch, and the sign-out every page's log-out button triggers — is bounded
 * inside the module that owns it, with no `base` at all: one card unmounting
 * must not take the rest of the view's request down with it, so a timeout is
 * the only thing that may abort it.
 *
 * `AbortSignal.timeout` would cover the plain "abort after N ms" case, and
 * vitest's fake timers do drive it here. It is not used because it cannot be
 * disarmed and cannot be told apart from a caller's own abort: this helper
 * needs `clear()` to stop a timer for a request that already finished, and
 * `timedOut()` to report a timeout without mislabelling an unmount.
 */

/** `/recommend/*` is the one backend path that degrades silently. */
export const RECOMMEND_DEADLINE_MS = 15_000
/** Chat session CRUD and the non-stream send: a full LLM round trip. */
export const CHAT_API_DEADLINE_MS = 30_000
/**
 * `/api/auth/me`, polled every 30 s by the analytics dashboard.
 *
 * Must stay ABOVE the dashboard's own 10 s identity race. `getMe` reports a
 * transport failure by rethrowing, and the dashboard's catch ignores an
 * `AbortError` by design; if this deadline fired first it would win that race
 * with exactly that error, and the user would get no message at all — the same
 * silent-hang symptom this helper exists to remove. At 15 s the dashboard's
 * race always settles first and reports "identity check timed out", and the
 * request is cancelled by the caller aborting its own signal, so nothing is
 * left in flight once the caller has stopped caring.
 */
export const ME_DEADLINE_MS = 15_000
/**
 * The sign-out POST. Fire-and-forget behind an immediate redirect, so nobody
 * is waiting on the answer — the bound only exists to release the socket.
 * Short on purpose: the user has already left the page, and the request is
 * only worth keeping alive for as long as a normal auth round trip takes.
 */
export const LOGOUT_DEADLINE_MS = 10_000

export interface Deadline {
  /** Pass to `fetch({ signal })`. Aborts on the deadline or on `base`. */
  signal: AbortSignal
  /**
   * True only if the DEADLINE fired. A caller-initiated abort (unmount) leaves
   * this false, so a timeout can be reported to the user while an unmount
   * stays silent.
   */
  timedOut(): boolean
  /** Stop the timer and detach the `base` listener. Always call on settle. */
  clear(): void
}

/**
 * An abort signal that fires after `ms`, composed with an optional caller
 * signal (`base`, e.g. an unmount controller). `base` aborting propagates
 * immediately and does not set `timedOut()`.
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

/**
 * A request that exceeded its deadline. Distinct from a transport
 * `AbortError`, so call sites can tell "the backend was too slow" from "this
 * request was cancelled" (unmount) from a plain network failure.
 */
export class RequestTimeoutError extends Error {
  readonly ms: number

  constructor(ms: number) {
    super(`The request timed out after ${Math.round(ms / 1000)}s. Check your connection and try again.`)
    this.name = 'RequestTimeoutError'
    this.ms = ms
  }
}
