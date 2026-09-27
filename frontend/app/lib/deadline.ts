/**
 * Request deadlines for the `fetch` call sites on the pages this helper
 * governs: the For You feed, SimilarArticles, chat's JSON `api()` helper and
 * `getMe()` — plus the two fire-and-forget beacons in those same files
 * (click tracking, logout) that would otherwise hold a socket open with
 * nothing to release it.
 *
 * A backend that accepts the connection but never answers leaves a promise
 * pending forever: no rejection, so every `catch`/`finally` downstream is never
 * reached and any `loading` flag stays true — the user gets a spinner with no
 * error and no way out. A deadline that aborts the underlying socket (rather
 * than merely abandoning the promise) is what turns that hang into a failure
 * the UI can report.
 *
 * NOT in scope: the search page, login, signup and the dashboard's own
 * analytics fetches keep their existing inline `setTimeout(() => abort())`
 * deadlines, and the chat SSE stream keeps its separate 45 s watchdog. This
 * module is the budget set for the pages listed above, not a claim that every
 * fetch in the app is routed through here.
 *
 * The pattern lives here as a small signal-level helper rather than a `fetch`
 * wrapper: each call site keeps its own `AbortController` for unmount
 * cancellation and passes that signal in as `base`, so unmount and deadline
 * remain independently observable (call sites gate their state updates on
 * `base.aborted`, not on the composed signal).
 *
 * `AbortSignal.timeout` is intentionally not used: its timer lives outside
 * `setTimeout`, so fake timers in tests cannot drive it.
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
