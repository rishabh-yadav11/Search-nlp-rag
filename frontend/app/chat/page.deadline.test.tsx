/**
 * A hung non-SSE `api()` call left `send()`'s first-turn await pending forever: the typing indicator
 * never cleared and the composer stayed blocked by `sendingRef`, with no error and no way to retry.
 * The stub below hangs only the non-SSE endpoints; the SSE stream keeps its own 45 s watchdog.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CHAT_API_DEADLINE_MS } from '../lib/deadline'
import { clearMeCache } from '../lib/auth'
import ChatPage from './page'

/**
 * Every fetch the page issues, with its signal. Calls are selected by endpoint rather than position,
 * because the mount-time `/api/auth/me` identity check (the cookie is httpOnly, so it is the only way
 * to tell a signed-in visitor from a signed-out one) means a call added at the front could otherwise
 * make a test quietly re-test its neighbour.
 */
let calls: { url: string; method: string; signal: AbortSignal | null }[] = []

/** The first (and only) call to `path` with `method`, or a test failure. */
function callTo(path: string, method = 'GET') {
  const found = calls.find((c) => c.url.includes(path) && c.method === method)
  if (!found) {
    throw new Error(
      `no ${method} ${path} among the page's calls: ` +
        calls.map((c) => `${c.method} ${c.url}`).join(', ')
    )
  }
  return found
}

type StubResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
  text: () => Promise<string>
}

function jsonResponse(data: unknown): StubResponse {
  return { ok: true, status: 200, json: async () => data, text: async () => JSON.stringify(data) }
}

function hangingApi() {
  return vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const signal = init?.signal ?? null
    calls.push({ url: String(input), method: init?.method ?? 'GET', signal })
    const { promise, reject } = Promise.withResolvers<StubResponse>()
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

beforeEach(() => {
  vi.useFakeTimers()
  calls = []
  Element.prototype.scrollTo = function scrollTo() {}
  // `getMe` memoises the session in module state for 60s and only `clearMeCache()` resets it, while
  // vitest shares one module registry across this file: without this an earlier test's
  // `/api/auth/me` answer would satisfy the later ones from cache.
  clearMeCache()
  vi.stubGlobal('fetch', hangingApi())
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

/** Submit a first turn, which POSTs `/api/chat/sessions` before streaming. */
async function submitFirstTurn(question: string) {
  render(<ChatPage />)
  fireEvent.change(screen.getByLabelText('Message'), { target: { value: question } })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  })
}

describe('ChatPage — api() calls are deadline-bounded', () => {
  it('aborts the hung session create and reports the failure instead of spinning', async () => {
    await submitFirstTurn('what happened to the deal?')

    const createCall = callTo('/api/chat/sessions', 'POST')
    expect(createCall.signal).toBeTruthy()
    expect(createCall.signal?.aborted).toBe(false)

    await advance(CHAT_API_DEADLINE_MS)

    expect(createCall.signal?.aborted).toBe(true)
    // The deadline's own message replaces the generic one, so a timeout is distinguishable.
    expect(screen.getByRole('alert').textContent).toMatch(/timed out after 30s/)
  })

  it('clears the in-flight guard so the user can retry after a timeout', async () => {
    await submitFirstTurn('what happened to the deal?')
    await advance(CHAT_API_DEADLINE_MS)

    const before = calls.length
    await submitFirstTurnAgain('trying once more')
    expect(calls.length).toBeGreaterThan(before)
  })

  it('aborts the hung sidebar session list at the deadline', async () => {
    render(<ChatPage />)
    const listCall = callTo('/api/chat/sessions', 'GET')
    expect(listCall.signal?.aborted).toBe(false)

    await advance(CHAT_API_DEADLINE_MS)

    expect(listCall.signal?.aborted).toBe(true)
  })

  it('reports a failed delete instead of leaking an unhandled rejection', async () => {
    const deleteSignals: (AbortSignal | null)[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.includes('/api/chat/sessions/') && init?.method === 'DELETE') {
          deleteSignals.push(init.signal ?? null)
          const { promise, reject } = Promise.withResolvers<StubResponse>()
          init.signal?.addEventListener('abort', () => {
            const err = new Error('The operation was aborted')
            err.name = 'AbortError'
            reject(err)
          })
          return promise
        }
        if (url.endsWith('/api/chat/sessions')) {
          return Promise.resolve(jsonResponse([{ id: 's1', title: 'Budget', created_at: 1_700_000_000, updated_at: 1_700_000_100 }]))
        }
        return Promise.resolve(jsonResponse({}))
      })
    )

    render(<ChatPage />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const del = screen.getAllByText('✕')[0]
    await act(async () => {
      fireEvent.click(del)
    })

    expect(deleteSignals).toHaveLength(1)
    await advance(CHAT_API_DEADLINE_MS)

    // The delete handler has no rethrow of its own, so this must land in the UI.
    expect(screen.getByRole('alert').textContent).toMatch(/timed out after 30s/)
  })
})

describe('ChatPage — the logout beacon is bounded too', () => {
  it('aborts a hung logout POST instead of leaving the socket open', async () => {
    const captured: { signal: AbortSignal | null } = { signal: null }
    const hangOthers = hangingApi()
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).includes('/api/auth/logout')) {
          captured.signal = init?.signal ?? null
          const { promise, reject } = Promise.withResolvers<StubResponse>()
          init?.signal?.addEventListener('abort', () => {
            const err = new Error('The operation was aborted')
            err.name = 'AbortError'
            reject(err)
          })
          return promise
        }
        // The sign-out control lives in the shared top bar and renders only once the identity check
        // answers; hanging that would leave nothing to click.
        if (String(input).includes('/api/auth/me')) {
          return Promise.resolve(
            jsonResponse({ id: 'u1', email: 'user@example.com', name: 'User', role: 'user', is_active: true })
          )
        }
        return hangOthers(input, init)
      })
    )

    render(<ChatPage />)
    await advance(0)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Log out' }))
    })

    expect(captured.signal).toBeTruthy()
    expect(captured.signal?.aborted).toBe(false)

    await advance(CHAT_API_DEADLINE_MS)
    expect(captured.signal?.aborted).toBe(true)
  })
})

/** Re-render into a fresh send on the already-rendered page. */
async function submitFirstTurnAgain(question: string) {
  fireEvent.change(screen.getByLabelText('Message'), { target: { value: question } })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  })
}
