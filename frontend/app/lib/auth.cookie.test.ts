/** The session is an httpOnly cookie: JS can never read it, and it only reaches the backend when we send `credentials: 'include'` to a trusted API base. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type * as AuthTypes from './auth'

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

let fetchInits: RequestInit[] = []
let fetchUrls: string[] = []

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

// The `/me` cache is module scope, so cache tests need a FRESH copy of the module.
async function freshAuth(): Promise<AuthModule> {
  vi.resetModules()
  return import('./auth')
}

describe('session cookie replaces the localStorage token', () => {
  it('leaves no usable credential in localStorage after a successful login', async () => {
    // Worst case: a working session is already sitting in storage when the user signs in.
    localStorage.setItem(LEGACY_KEY, 'stale-jwt-from-old-build')
    const auth = await freshAuth()

    route = () =>
      jsonResponse({
        // Token-looking fields the client must ignore entirely.
        user: USER_A,
        access_token: 'SHOULD-NEVER-BE-STORED',
        token: 'SHOULD-NEVER-BE-STORED',
      })

    // Mirrors the login page: POST, then drop the legacy token and reset the cache.
    const res = await fetch('/api/auth/login', auth.authRequestInit({ method: 'POST' }))
    const body = (await res.json()) as Record<string, unknown>
    auth.clearMeCache()
    auth.clearLegacyToken()
    route = () => jsonResponse(USER_A)
    await auth.getMe()

    expect(body.user).toEqual(USER_A)
    expect(localStorage.getItem(LEGACY_KEY)).toBeNull()
    expect(localStorage.length).toBe(0)
    expect(JSON.stringify({ ...localStorage })).not.toContain('SHOULD-NEVER-BE-STORED')
  })

  it('exposes no API that can write a credential to storage', async () => {
    const auth = await freshAuth()
    // getToken/setToken/clearToken together were a full write/read API for the session.
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
    expect(init.credentials).toBe('include')
    // The backend takes no bearer tokens and JS cannot read the cookie, so there must be no Authorization header.
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

    // Resolving null would bounce a signed-in user to /login on a flaky connection.
    await expect(getMe()).rejects.toBe(boom)
  })

  it('keeps serving a cached user on a network failure rather than logging out', async () => {
    const { getMe } = await freshAuth()
    route = () => jsonResponse(USER_A)
    expect(await getMe()).toEqual(USER_A)

    // Cache warm, backend unreachable: the cached identity must not be discarded.
    route = () => {
      throw new TypeError('Failed to fetch')
    }
    expect(await getMe()).toEqual(USER_A)
  })
})

describe('/me cache isolation between users', () => {
  it('does not serve user A after user B logs in on the same tab', async () => {
    const { getMe, clearMeCache } = await freshAuth()

    route = () => jsonResponse(USER_A)
    expect(await getMe()).toEqual(USER_A)
    expect(await getMe()).toEqual(USER_A) // served from cache, no second fetch
    expect(fetchUrls).toHaveLength(1)

    // B signs in on the same tab: clearing the cache must send the next read back to the server.
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

    // A real 401 must invalidate the cache, not just return null once; `force` gets past the warm cache.
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
    // The env must be stubbed before the import: the base is resolved once at module scope.
    vi.stubEnv('NEXT_PUBLIC_API_BASE', 'http://api.example.com')
    const { authRequestInit, API_BASE_TRUSTED, API_BASE } = await freshAuth()

    expect(API_BASE_TRUSTED).toBe(false)
    // `baseFromUrl` preserves the base's path, hence the trailing slash.
    expect(API_BASE).toBe('http://api.example.com/')

    // Returned unchanged, so no `credentials` key can leak the cookie cross-origin.
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
    // keepalive is load-bearing: the redirect tears down an in-flight fetch, and an httpOnly cookie
    // cannot be cleared by script, so a cancelled POST leaves the session live.
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
    expect(logoutCall?.credentials).toBe('include')
    expect(new Headers(logoutCall?.headers).has('Authorization')).toBe(false)
  })

  it('still navigates away when the revocation request fails', async () => {
    // A network failure must not strand the user on an authenticated page.
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
