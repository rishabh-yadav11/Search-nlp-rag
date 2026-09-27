/**
 * Request deadlines for the frontend's `fetch` call sites.
 *
 * A backend that accepts the connection but never answers leaves a promise
 * pending forever: no rejection, so every `catch`/`finally` downstream is never
 * reached and any `loading` flag stays true — the user gets a spinner with no
 * error and no way out. Every request the browser issues therefore has to carry
 * a deadline that aborts the underlying socket, not just a timeout that
 * abandons the promise.
 *
 * The per-page `setTimeout(() => controller.abort(), MS)` idiom was open-coded
 * at five call sites with four different budgets; a sixth through ninth copy
 * would make the envelope impossible to audit in one place, so the pattern
 * lives here as a small signal-level helper. It is deliberately NOT a `fetch`
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
/** `/api/auth/me`, polled every 30 s by the analytics dashboard. */
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
