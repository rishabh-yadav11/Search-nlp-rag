'use client'

export const TOKEN_KEY = 'vccircle_auth_token'

export const API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000')

export function getToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string): void {
  try {
    window.localStorage.setItem(TOKEN_KEY, token)
  } catch {
    /* storage unavailable */
  }
}

export function clearToken(): void {
  try {
    window.localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage unavailable */
  }
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

/** Fetch the current authenticated user (`/api/auth/me`), cached per token.
 *  Returns null when logged out or the token is rejected. Never redirects. */
export async function getMe(force = false): Promise<AuthUser | null> {
  const token = getToken()
  if (!token) {
    meCache = null
    meCacheToken = null
    return null
  }
  if (!force && meCache !== undefined && meCacheToken === token) return meCache
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders() })
    if (res.status === 401) {
      clearToken()
      meCache = null
      meCacheToken = null
      return null
    }
    if (!res.ok) return null
    meCache = (await res.json()) as AuthUser
    meCacheToken = token
    return meCache
  } catch {
    return null
  }
}

export function clearMeCache(): void {
  meCache = undefined
  meCacheToken = null
}

/** Redirect to the login page (used when the backend rejects an expired token).
 *  `next` (a path) is preserved so the user is sent back after signing in. */
export function redirectToLogin(next?: string): void {
  clearToken()
  clearMeCache()
  if (typeof window !== 'undefined') {
    const target = next && next.startsWith('/') ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}