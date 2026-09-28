/**
 * Issue #247 — the session is an httpOnly cookie, not a token in localStorage.
 *
 * These tests pin the two properties the migration has to hold:
 *  1. JavaScript can never read the session, so nothing readable is left in
 *     storage and no request carries an `Authorization` header; and
 *  2. the httpOnly cookie only reaches the backend when we ask for it
 *     (`credentials: 'include'`), and only when the API base is trusted.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as AuthTypes from './auth'

/** The parts of the auth module these tests drive. */
type AuthModule = Pick<
  typeof AuthTypes,
  | 'API_BASE'
  | 'API_BASE_TRUSTED'
  | 'authRequestInit'
  | 'clearLegacyToken'
  | 'clearMeCache'
  | 'getMe'
  | 'logout'
>

const LEGACY_KEY = 'vccircle_auth_token'

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

const USER_A = { id: 'a1', email: 'a@example.com', name: 'A', role: 'user', is_active: true }
const USER_B = { id: 'b1', email: 'b@example.com', name: 'B', role: 'user', is_active: true }

/** Every RequestInit the module handed to fetch, in order. */
let fetchInits: RequestInit[] = []
/** Every URL the module fetched, in order. */
let fetchUrls: string[] = []

/** Route a single URL to a response; anything else 404s. */
let route: (url: string, init: RequestInit) => StubResponse

beforeEach(() => {
  fetchInits = []
  fetchUrls = []
  route = () => jsonResponse({ detail: 'not found' }, 404)
  localStorage.clear()
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const resolved = init ?? {}
      fetchUrls.push(url)
      fetchInits.push(resolved)
      return route(url, resolved)
    })
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.resetModules()
})

// The module keeps a time-keyed `/me` cache at module scope, so every test that
// asserts on cache behaviour imports a FRESH copy of the module.
async function freshAuth(): Promise<AuthModule> {
  vi.resetModules()
  return import('./auth')
}

describe('session cookie replaces the localStorage token', () => {
  it('leaves no usable credential in localStorage after a successful login', async () => {
    // A pre-cookie build left a readable token behind: the worst case, because
    // a working session is already sitting in storage when the user signs in.
    localStorage.setItem(LEGACY_KEY, 'stale-jwt-from-old-build')
    const auth = await freshAuth()

    route = () =>
      jsonResponse({
        // The backend returns the user, not a token. The stub deliberately
        // includes token-looking fields the client must ignore entirely.
        user: USER_A,
        access_token: 'SHOULD-NEVER-BE-STORED',
        token: 'SHOULD-NEVER-BE-STORED',
      })

    // Exactly what the login page does: POST the login, read the body, then
    // reset the cached identity and drop the legacy key.
    const res = await fetch('/api/auth/login', auth.authRequestInit({ method: 'POST' }))
    const body = (await res.json()) as Record<string, unknown>
    auth.clearMeCache()
    auth.clearLegacyToken()
    route = () => jsonResponse(USER_A)
    await auth.getMe()

    expect(body.user).toEqual(USER_A)
    // The old key is gone...
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
    // ...and nothing replaced it: no key of any name holds a credential, and in
    // particular no response field was persisted.
    expect(localStorage.length).toBe(0)
    expect(JSON.stringify({ ...localStorage })).not.toContain('SHOULD-NEVER-BE-STORED')
  })

  it('exposes no API that can write a credential to storage', async () => {
    const auth = await freshAuth()
    // The old module exported getToken/setToken/clearToken, which together were
    // a complete write/read API for the session. None of that may come back.
    const exported = Object.keys(auth) as Array<keyof typeof auth>
    expect(exported).not.toContain('setToken' as keyof typeof auth)
    expect(exported).not.toContain('getToken' as keyof typeof auth)
    expect(exported).not.toContain('clearToken' as keyof typeof auth)
    expect(exported).not.toContain('TOKEN_KEY' as keyof typeof auth)
  })

  it('removes a legacy vccircle_auth_token left by a pre-cookie build', async () => {
    localStorage.setItem(LEGACY_KEY, 'stale-jwt-from-old-build')
    const { getMe } = await freshAuth()

    route = () => jsonResponse(USER_A)
    const me = await getMe()

    expect(me).toEqual(USER_A)
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
  })

  it('removes the legacy token even when the session is rejected (401)', async () => {
    localStorage.setItem(LEGACY_KEY, 'stale-jwt-from-old-build')
    const { getMe } = await freshAuth()

    route = () => jsonResponse({ detail: 'not authenticated' }, 401)
    expect(await getMe()).toBeNull()

    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
  })
})

describe('getMe', () => {
  it('sends credentials and no Authorization header', async () => {
    const { getMe } = await freshAuth()
    route = () => jsonResponse(USER_A)

    await getMe()

    expect(fetchInits).toHaveLength(1)
    const init = fetchInits[0]
    // The cookie rides along only because we asked for it.
    expect(init.credentials).toBe('include')
    // The backend no longer accepts bearer tokens, and JS cannot read the
    // cookie, so there must be no Authorization header at all.
    expect(new Headers(init.headers).has('Authorization')).toBe(false)
    expect(fetchUrls[0]).toContain('/api/auth/me')
  })

  it('returns null on 401 — a definitive logged-out state', async () => {
    const { getMe } = await freshAuth()
    route = () => jsonResponse({ detail: 'expired' }, 401)

    expect(await getMe()).toBeNull()
  })

  it('rethrows on a network failure instead of pretending the user logged out', async () => {
    const { getMe } = await freshAuth()
    const boom = new TypeError('Failed to fetch')
    route = () => {
      throw boom
    }

    // A transient network error is NOT a logout. Resolving null here would make
    // every page redirect a signed-in user to /login on a flaky connection.
    await expect(getMe()).rejects.toBe(boom)
  })

  it('keeps serving a cached user on a network failure rather than logging out', async () => {
    const { getMe } = await freshAuth()
    route = () => jsonResponse(USER_A)
    expect(await getMe()).toEqual(USER_A)

    // Cache warm, backend now unreachable: the cached identity is still valid
    // and must not be discarded as a logout.
    route = () => {
      throw new TypeError('Failed to fetch')
    }
    expect(await getMe()).toEqual(USER_A)
  })
})

describe('/me cache isolation between users', () => {
  it('does not serve user A after user B logs in on the same tab', async () => {
    const { getMe, clearMeCache } = await freshAuth()

    // User A is signed in; the cache warms with A's record.
    route = () => jsonResponse(USER_A)
    expect(await getMe()).toEqual(USER_A)
    expect(await getMe()).toEqual(USER_A) // served from cache, no second fetch
    expect(fetchUrls).toHaveLength(1)

    // B signs in on the same tab. The login success path clears the cache, so
    // the next read must go back to the server instead of replaying A.
    route = () => jsonResponse(USER_B)
    clearMeCache()

    const me = await getMe()
    expect(me).toEqual(USER_B)
    expect(fetchUrls).toHaveLength(2)
    expect(me?.email).not.toBe(USER_A.email)
  })

  it('drops the cached user after a 401 so the next read re-checks the server', async () => {
    const { getMe } = await freshAuth()

    route = () => jsonResponse(USER_A)
    expect(await getMe()).toEqual(USER_A)

    // Session expires: a real 401 must invalidate the cache, not just return
    // null once. `force` is what gets past the warm cache to reach the server.
    route = () => jsonResponse({ detail: 'expired' }, 401)
    expect(await getMe(true)).toBeNull()

    route = () => jsonResponse(USER_B)
    expect(await getMe()).toEqual(USER_B)
    expect(fetchUrls).toHaveLength(3)
  })

  it('refetches when force is set, even inside the TTL', async () => {
    const { getMe } = await freshAuth()
    route = () => jsonResponse(USER_A)

    await getMe()
    await getMe(true)
    expect(fetchUrls).toHaveLength(2)
  })
})

describe('authRequestInit', () => {
  it('adds credentials: include to a trusted base while preserving the caller init', async () => {
    const { authRequestInit, API_BASE_TRUSTED } = await freshAuth()
    expect(API_BASE_TRUSTED).toBe(true) // empty base → same-origin, trusted

    const signal = new AbortController().signal
    const init = authRequestInit({ method: 'POST', headers: { 'Content-Type': 'application/json' }, signal })

    expect(init.credentials).toBe('include')
    expect(init.method).toBe('POST')
    expect(init.signal).toBe(signal)
    expect(new Headers(init.headers).get('Content-Type')).toBe('application/json')
  })

  it('returns an empty init with no credentials when no init is passed', async () => {
    const { authRequestInit } = await freshAuth()
    expect(authRequestInit()).toEqual({ credentials: 'include' })
  })

  it('omits credentials entirely when the API base is untrusted', async () => {
    // A configured cross-origin http base that is neither loopback nor
    // allow-listed is the one input that yields an untrusted base. It must be
    // stubbed before the module is imported, since the value is resolved once
    // at module scope.
    vi.stubEnv('NEXT_PUBLIC_API_BASE', 'http://api.example.com')
    const { authRequestInit, API_BASE_TRUSTED, API_BASE } = await freshAuth()

    expect(API_BASE_TRUSTED).toBe(false)
    // `baseFromUrl` preserves the base's path, hence the trailing slash.
    expect(API_BASE).toBe('http://api.example.com/')

    // Returned unchanged: the caller gets back exactly what it passed in, with
    // no `credentials` key that would leak the cookie cross-origin.
    const callerInit: RequestInit = { method: 'POST' }
    const init = authRequestInit(callerInit)
    expect(init).toBe(callerInit)
    expect('credentials' in init).toBe(false)
    expect(init.credentials).toBeUndefined()
    expect(authRequestInit()).toEqual({})
  })

  it('keeps getMe from sending credentials to an untrusted base', async () => {
    vi.stubEnv('NEXT_PUBLIC_API_BASE', 'http://api.example.com')
    const { getMe } = await freshAuth()

    route = () => jsonResponse(USER_A)
    await getMe()

    expect(fetchUrls[0]).toBe('http://api.example.com//api/auth/me')
    expect(fetchInits[0].credentials).toBeUndefined()
  })

  it('still sends credentials to an allow-listed cross-origin https base', async () => {
    vi.stubEnv('NEXT_PUBLIC_API_BASE', 'https://api.example.com')
    vi.stubEnv('NEXT_PUBLIC_TRUSTED_API_HOSTS', 'api.example.com')
    const { getMe, API_BASE_TRUSTED } = await freshAuth()

    expect(API_BASE_TRUSTED).toBe(true)
    route = () => jsonResponse(USER_A)
    await getMe()

    expect(fetchInits[0].credentials).toBe('include')
  })
})

describe('logout', () => {
  it('sends the revocation with keepalive so it survives the redirect', async () => {
    // The reason this is a test and not a comment: `redirectToLogin` navigates
    // away on the next line, which tears down an ordinary in-flight fetch. An
    // httpOnly cookie cannot be cleared by script, so a cancelled POST leaves
    // the session live on the server and in the browser -- the user returns to
    // /login still authenticated, and a shared machine stays signed in.
    const { logout } = await freshAuth()
    const replaced = vi.fn()
    const original = window.location
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...original, replace: replaced, origin: 'http://localhost:3000' },
    })

    try {
      route = () => jsonResponse({ ok: true })
      logout()
    } finally {
      Object.defineProperty(window, 'location', { configurable: true, value: original })
    }

    expect(replaced).toHaveBeenCalled()
    const logoutCall = fetchInits.find((_, i) => fetchUrls[i].endsWith('/api/auth/logout'))
    expect(logoutCall).toBeDefined()
    expect(logoutCall?.keepalive).toBe(true)
    // Credentialed, so the cookie rides and the server can revoke the session.
    expect(logoutCall?.credentials).toBe('include')
    // And still nothing script-readable is attached to it.
    expect(new Headers(logoutCall?.headers).has('Authorization')).toBe(false)
  })

  it('still navigates away when the revocation request fails', async () => {
    // The page must not strand the user on an authenticated page because the
    // network is down; the session simply dies with the tab.
    const { logout } = await freshAuth()
    const replaced = vi.fn()
    const original = window.location
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...original, replace: replaced, origin: 'http://localhost:3000' },
    })
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down')
      })
    )

    try {
      logout()
      await Promise.resolve()
    } finally {
      Object.defineProperty(window, 'location', { configurable: true, value: original })
    }

    expect(replaced).toHaveBeenCalled()
  })
})
