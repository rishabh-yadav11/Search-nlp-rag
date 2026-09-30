/**
 * Issue #287 — chat's JSON `api()` helper had no signal at all, so a hung
 * session-list or session-create call left `send()`'s first-turn await pending
 * forever: the typing indicator never went away, the composer stayed blocked by
 * `sendingRef`, and the user got no error and no way to retry.
 *
 * The stub below hangs only the non-SSE endpoints; the SSE stream keeps its
 * existing 45 s watchdog and is out of scope here.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CHAT_API_DEADLINE_MS } from '../lib/deadline'
import { clearMeCache } from '../lib/auth'
import ChatPage from './page'

/**
 * Every fetch the page issues, with the signal it was given. The page's mount
 * now opens with a `/api/auth/me` identity check — the cookie is httpOnly, so
 * that request is the only way to tell a signed-in visitor from a signed-out
 * one — which means the calls can no longer be told apart by position. They
 * are selected by endpoint instead, so a call added at the front cannot make a
 * test quietly re-test its neighbour.
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

/** Every chat API call hangs until its signal aborts. */
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
  // `getMe` memoises the session in module state for 60s and only
  // `clearMeCache()` resets it, and vitest shares one module registry across
  // every test in this file. An earlier test whose stub answers
  // `/api/auth/me` would otherwise satisfy the later ones from cache, so the
  // per-test stub below would never be reached.
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

    // The mount-time session LIST and the session CREATE both go to
    // /api/chat/sessions; only the method tells them apart.
    const createCall = callTo('/api/chat/sessions', 'POST')
    expect(createCall.signal).toBeTruthy()
    expect(createCall.signal?.aborted).toBe(false)

    await advance(CHAT_API_DEADLINE_MS)

    expect(createCall.signal?.aborted).toBe(true)
    // The caller's generic copy is replaced by the deadline's own message, so a
    // timeout is distinguishable from any other failure the user can retry.
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
    // The session list resolves so a row exists to delete; the DELETE hangs.
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

    // The delete handler has no rethrow of its own, so this must land in the
    // UI. An uncaught rejection here would fail the run instead.
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
        // The sign-out control now lives in the shared top bar, which renders
        // it only once the page's own identity check has answered. Hanging this
        // one would leave nothing to click, so sign the visitor in.
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
