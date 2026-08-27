'use client'

export const TOKEN_KEY = 'vccircle_auth_token'

// Hosts explicitly allowed to receive the auth Bearer token. This only matters
// for a runtime-injected `window.API_BASE` (see below); build-time config is
// operator-controlled and trusted. Set NEXT_PUBLIC_TRUSTED_API_HOSTS to a
// comma-separated list (e.g. "api.example.com") to permit runtime overrides to
// a known-good backend.
const TRUSTED_API_HOSTS: string[] = (process.env.NEXT_PUBLIC_TRUSTED_API_HOSTS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)

// Candidates, in priority order. `window.API_BASE` is runtime/injectable (e.g.
// via an XSS payload or a malicious inline script) and therefore untrusted
// unless its host is explicitly allow-listed — it is the only attacker-reachable
// vector here. The build-time env var and the dev default are operator-controlled
// and treated as trusted.
const WIN_API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) || ''
const ENV_API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''
const DEV_API_BASE = process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : ''

/**
 * Validate a candidate API base. Rejects anything that is not a proper http(s)
 * URL: no `javascript:`/other schemes, no protocol-relative URLs, no embedded
 * credentials, and no whitespace that could cause host confusion. Returns the
 * normalized origin (trusted flag set by the caller based on source).
 */
function parseApiBase(value: string): URL | null {
  if (!value || /\s/.test(value)) return null
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return null
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null
  // Embedded userinfo (e.g. `https://user@host`) is a red flag — reject.
  if (url.username || url.password) return null
  return url
}

function resolveApiBase(): { base: string; trusted: boolean } {
  // Runtime-injected base: only trusted if its host is allow-listed.
  if (WIN_API_BASE) {
    const url = parseApiBase(WIN_API_BASE)
    if (!url) {
      console.error(
        '[auth] window.API_BASE is not a valid http(s) URL; ignoring it and falling back to a safe same-origin base (auth token will not be sent cross-origin).'
      )
      return { base: '', trusted: false }
    }
    if (TRUSTED_API_HOSTS.includes(url.host)) {
      return { base: url.origin, trusted: true }
    }
    console.error(
      `[auth] window.API_BASE host "${url.host}" is not in NEXT_PUBLIC_TRUSTED_API_HOSTS; falling back to a safe same-origin base (auth token will not be sent cross-origin).`
    )
    return { base: '', trusted: false }
  }
  // Operator-controlled build-time config / dev default: trusted.
  const trustedSource = ENV_API_BASE || DEV_API_BASE
  if (trustedSource) {
    const url = parseApiBase(trustedSource)
    if (!url) {
      console.error(
        `[auth] Configured API base "${trustedSource}" is not a valid http(s) URL; falling back to a safe same-origin base.`
      )
      return { base: '', trusted: false }
    }
    return { base: url.origin, trusted: true }
  }
  // Production: no base set → same-origin relative requests (safe, token stays
  // first-party). NEXT_PUBLIC_API_BASE must be set at build for a real backend.
  return { base: '', trusted: false }
}

const RESOLVED_API_BASE = resolveApiBase()

/** The validated API base. Empty string means same-origin relative requests. */
export const API_BASE = RESOLVED_API_BASE.base
/** True only when API_BASE is a trusted backend allowed to receive the token. */
export const API_BASE_TRUSTED = RESOLVED_API_BASE.trusted

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
  // SECURITY: localStorage is XSS-exfiltratable — any script on the page can
  // read this token, so a single XSS easily steals the session. The correct fix
  // is a backend-set httpOnly + Secure + SameSite cookie (backend change, out of
  // scope). This comment marks the token-storage site: the token MUST move to an
  // httpOnly cookie. Until then, NEVER write the token value to logs, console,
  // errors, analytics, or any outbound payload. This path is a known risk.
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
  // SECURITY: the Bearer token is only attached when API_BASE is a trusted
  // backend (same-origin or an explicitly allow-listed host). Attaching it to
  // an attacker-controlled/untrusted base would leak the credential
  // cross-origin. See API_BASE_TRUSTED / resolveApiBase above. If the base is
  // untrusted, requests are still sent (unauthenticated) but the token never
  // leaves the first party.
  if (token && API_BASE_TRUSTED) headers.set('Authorization', `Bearer ${token}`)
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