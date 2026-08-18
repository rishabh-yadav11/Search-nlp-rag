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

/** Redirect to the login page (used when the backend rejects an expired token).
 *  `next` (a path) is preserved so the user is sent back after signing in. */
export function redirectToLogin(next?: string): void {
  clearToken()
  if (typeof window !== 'undefined') {
    const target = next && next.startsWith('/') ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}