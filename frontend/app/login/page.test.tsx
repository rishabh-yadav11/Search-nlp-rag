/**
 * Issue #247 — the login page must not persist a credential.
 *
 * The backend answers a successful login with an httpOnly cookie and a user
 * object. This pins the client half of that contract: nothing readable is
 * written to storage, the request is sent so the cookie can be set, and the
 * user is redirected onward.
 */
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LoginPage from './page'

const LEGACY_KEY = 'vccircle_auth_token'

const replace = vi.hoisted(() => vi.fn())

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(''),
}))

type StubResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
  text: () => Promise<string>
}

function jsonResponse(data: unknown, status = 200): StubResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  }
}

const ME_USER = { id: 'u1', email: 'a@example.com', name: 'A', role: 'user', is_active: true }

let loginInits: RequestInit[] = []

function stubFetch(loginResponse: () => StubResponse) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<StubResponse> => {
      const url = String(input)
      if (url.endsWith('/api/auth/login')) {
        loginInits.push(init ?? {})
        return loginResponse()
      }
      // Not signed in yet, so the mount check must leave the form alone.
      if (url.endsWith('/api/auth/me')) return jsonResponse({ detail: 'not authenticated' }, 401)
      return jsonResponse({ detail: 'not found' }, 404)
    })
  )
}

beforeEach(() => {
  replace.mockClear()
  loginInits = []
  localStorage.clear()
  stubFetch(() => jsonResponse({ user: ME_USER }))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.resetModules()
})

async function submitLogin() {
  render(<LoginPage />)
  // Let the mount-time /api/auth/me check settle before interacting.
  await act(async () => {})
  fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'a@example.com' } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password123' } })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }))
  })
}

describe('LoginPage', () => {
  it('persists nothing and redirects to the app after a successful login', async () => {
    // A token from a previous, pre-cookie session is present on boot.
    localStorage.setItem(LEGACY_KEY, 'stale-jwt')

    await submitLogin()

    await waitFor(() => expect(replace).toHaveBeenCalledWith('/chat'))
    // Key-agnostic: NO entry at all may exist under any name, so a client that
    // invented a new storage key for the session would be caught here too.
    expect(localStorage.length).toBe(0)
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
  })

  it('sends the login request with credentials so the cookie can be set', async () => {
    await submitLogin()

    await waitFor(() => expect(loginInits).toHaveLength(1))
    expect(loginInits[0].credentials).toBe('include')
    expect(loginInits[0].method).toBe('POST')
    // The client has no bearer token to send and must not invent one.
    expect(new Headers(loginInits[0].headers).has('Authorization')).toBe(false)
  })

  it('stays on the login page and does not redirect when the session check 401s', async () => {
    // Mount only: the /api/auth/me check must find no session and leave the
    // user on the form rather than bouncing them somewhere.
    render(<LoginPage />)
    await act(async () => {})

    expect(screen.getByRole('button', { name: 'Sign in' })).toBeTruthy()
    expect(replace).not.toHaveBeenCalled()
  })

  it('shows an error and writes no credential when the login is rejected', async () => {
    localStorage.setItem(LEGACY_KEY, 'stale-jwt')
    stubFetch(() => jsonResponse({ detail: 'Invalid credentials' }, 401))

    await submitLogin()

    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Invalid credentials'))
    // A failed login must not navigate, and must not write a credential. The
    // mount-time /api/auth/me check has already deleted the pre-cookie token,
    // and a failed login adds nothing of its own.
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
    expect(localStorage.length).toBe(0)
  })
})
