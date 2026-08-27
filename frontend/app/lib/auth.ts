'use client'

export const TOKEN_KEY = 'vccircle_auth_token'

export const API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  // Only default to the local backend in development. In production an unset
  // base falls back to same-origin relative requests rather than leaking the
  // auth token to localhost:8000 (NEXT_PUBLIC_API_BASE must be set at build).
  (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '')

/**
 * Returns true only for a safe, same-origin, root-relative redirect path.
 * Safe means the value starts with exactly one `/` and is NOT:
 *  - a protocol-relative URL (e.g. `//evil.com`),
 *  - an absolute URL (e.g. `https://evil.com`),
 *  - an absolute path with an embedded scheme (any `:` before the first `/`
 *    or where the path begins with `//`).
 * Anything else is rejected and callers must fall back to a default like
 * `/chat` or `/`.
 */
export function isSafeRedirect(next: unknown): next is string {
  if (typeof next !== 'string' || next.length === 0) return false
  if (!next.startsWith('/')) return false
  // Reject protocol-relative URLs (`//evil.com`).
  if (next.startsWith('//')) return false
  // Reject any embedded scheme (`http:`, `javascript:`, etc.).
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(next)) return false
  return true
}

export function getToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string): void {
  // SECURITY: localStorage is XSS-exfiltratable. The token must move to an
  // httpOnly, Secure, SameSite cookie (backend change required). Never log or
  // echo the token value. This storage path is a known risk until that lands.
  try {
    window.localStorage.setItem(TOKEN_KEY, token)
  } catch {
    /* storage unavailable */
  }
  // Login (or re-login) may change role/is_active; drop any stale cached user.
  clearMeCache()
}

export function clearToken(): void {
  try {
    window.localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage unavailable */
  }
  // Logout invalidates the cached user so it is never served stale.
  clearMeCache()
}

export function authHeaders(init?: RequestInit): Headers {
  const headers = new Headers(init?.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return headers
}

export interface AuthUser {
  id: string
  email: string
  name: string
  role: string
  is_active: boolean
}

let meCache: AuthUser | null | undefined
let meCacheToken: string | null = null
let meCacheTs = 0

// Configurable TTL so role/is_active changes are eventually picked up even if
// the caller forgets to clear the cache. 0 disables the time-based expiry.
const ME_CACHE_TTL_MS = Number(process.env.NEXT_PUBLIC_ME_CACHE_TTL_MS || 60000)

/** Fetch the current authenticated user (`/api/auth/me`), cached per token.
 *  Returns null when the token is missing/rejected (401) or when the request
 *  fails (network/transport error or non-2xx). On a network/transport failure
 *  the token is preserved (not cleared) so a later retry can recover — callers
 *  must not treat a null return as a definitive "logged out" without also
 *  checking the token. Network failures are logged, not thrown, so existing
 *  `.catch` handlers don't misinterpret them as auth rejection. Never redirects. */
export async function getMe(force = false): Promise<AuthUser | null> {
  const token = getToken()
  if (!token) {
    clearToken()
    return null
  }
  const fresh = meCache !== undefined && meCacheToken === token && (ME_CACHE_TTL_MS <= 0 || Date.now() - meCacheTs < ME_CACHE_TTL_MS)
  if (!force && fresh) return meCache ?? null
  let res: Response
  try {
    res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders() })
  } catch (err) {
    // Network/transport failure: do NOT treat as "not authenticated" (preserve
    // the token so a later retry can succeed), but surface it rather than
    // swallowing it. Returning null here is graceful; callers that need the
    // underlying error can inspect console output.
    console.error('getMe: failed to reach the auth service', err)
    return null
  }
  if (res.status === 401) {
    clearToken()
    return null
  }
  if (!res.ok) return null
  try {
    meCache = (await res.json()) as AuthUser
  } catch {
    // Malformed/non-JSON 200 response: don't throw (callers may lack a
    // .catch); treat as an unexpected payload and return null safely.
    console.error('getMe: failed to parse /api/auth/me response')
    return null
  }
  meCacheToken = token
  meCacheTs = Date.now()
  return meCache
}

export function clearMeCache(): void {
  meCache = undefined
  meCacheToken = null
  meCacheTs = 0
}

/** Redirect to the login page (used when the backend rejects an expired token).
 *  `next` (a path) is preserved so the user is sent back after signing in. */
export function redirectToLogin(next?: string): void {
  clearToken()
  clearMeCache()
  if (typeof window !== 'undefined') {
    // Only preserve `next` when it is a safe, root-relative path. A value like
    // `//evil.com` or `https://evil.com` must fall back to a plain `/login`.
    const target = isSafeRedirect(next) ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}