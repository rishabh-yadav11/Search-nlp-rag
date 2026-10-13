


export const RECOMMEND_DEADLINE_MS = 15_000

export const CHAT_API_DEADLINE_MS = 30_000

export const ME_DEADLINE_MS = 15_000

export const LOGOUT_DEADLINE_MS = 10_000

export interface Deadline {
  
  signal: AbortSignal
  
  timedOut(): boolean
  
  clear(): void
}


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


export class RequestTimeoutError extends Error {
  readonly ms: number

  constructor(ms: number) {
    super(`The request timed out after ${Math.round(ms / 1000)}s. Check your connection and try again.`)
    this.name = 'RequestTimeoutError'
    this.ms = ms
  }
}
