'use client'
import { devApiBase, parseApiBaseUrl, readApiBaseEnv } from './api-base'
import { isSafeRedirect } from './safe-url'
import { createDeadline, LOGOUT_DEADLINE_MS, ME_DEADLINE_MS } from './deadline'

// Allow-listed hosts (comma-separated); a cross-origin base must be https AND listed here.
const TRUSTED_API_HOSTS: string[] = (process.env.NEXT_PUBLIC_TRUSTED_API_HOSTS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)

const ENV_API_BASE = readApiBaseEnv()
const DEV_API_BASE = devApiBase()


function isSameOrigin(url: URL): boolean {
  if (typeof window === 'undefined') return false
  return url.origin === window.location.origin
}

function isHttps(url: URL): boolean {
  return url.protocol === 'https:'
}

function isLoopback(url: URL): boolean {
  // `URL.hostname` returns IPv6 loopback without brackets (e.g. `::1`).
  return ['localhost', '127.0.0.1', '::1'].includes(url.hostname)
}

/** Normalize an allow-list entry or URL host so the two compare equal: `url.host` arrives
 *  lowercased and port-stripped, entries do not, so `API.Example.COM:443` would never match. */
export function normalizeHost(host: string): string {
  return host
    .replace(/^https?:\/\//, '')
    .replace(/\/+$/, '')
    .replace(/\.$/, '')
    .replace(/:(?:80|443)$/, '')
    .toLowerCase()
}

function hostInAllowList(url: URL): boolean {
  const h = normalizeHost(url.host)
  return TRUSTED_API_HOSTS.some((entry) => normalizeHost(entry) === h)
}

/** The request base keeping any path (`https://host/v1`), not just the origin. */
function baseFromUrl(url: URL): string {
  return url.origin + url.pathname
}

function isTrustedBase(base: string, url: URL | null): boolean {
  if (!base) return true // empty relative base → same-origin, first-party
  if (!url) return false
  if (isSameOrigin(url)) return true // same-origin (http ok for local dev)
  // Loopback is cross-origin but never a network cleartext risk, so http stays trusted.
  if (isLoopback(url)) return true
  return isHttps(url) && hostInAllowList(url)
}

function resolveApiBase(): { base: string; trusted: boolean } {
  // SSR has no current origin, so a relative base cannot resolve and falls back to same-origin.
  const currentOrigin = typeof window !== 'undefined' ? window.location.origin : undefined
  // `window.API_BASE` is the only attacker-reachable base (XSS), honoured only when
  // isTrustedBase agrees; read live so an override set after module load still applies.
  const winApiBase =
    (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) || ''
  if (winApiBase) {
    const url = parseApiBaseUrl(winApiBase, currentOrigin)
    if (!url) {
      console.error(
        '[auth] window.API_BASE is not a valid http(s) URL; ignoring it and using the safe same-origin base.'
      )
      return { base: '', trusted: true }
    }
    if (isTrustedBase(winApiBase, url)) {
      return { base: baseFromUrl(url), trusted: true }
    }
    console.error(
      `[auth] window.API_BASE host "${normalizeHost(url.host)}" is not same-origin and not in NEXT_PUBLIC_TRUSTED_API_HOSTS; using the safe same-origin base instead.`
    )
    return { base: '', trusted: true }
  }
  // Operator-controlled, but the same trust rule applies: cross-origin still needs https + an entry.
  const trustedSource = ENV_API_BASE || DEV_API_BASE
  if (trustedSource) {
    const url = parseApiBaseUrl(trustedSource, currentOrigin)
    if (!url) {
      console.error(
        `[auth] Configured API base "${trustedSource}" is not a valid http(s) URL; falling back to the safe same-origin base.`
      )
      return { base: '', trusted: true }
    }
    const trusted = isTrustedBase(trustedSource, url)
    if (!trusted) {
      console.error(
        `[auth] Configured API base "${trustedSource}" is cross-origin and not trusted (https + NEXT_PUBLIC_TRUSTED_API_HOSTS required); the session cookie will NOT be sent.`
      )
    }
    return { base: baseFromUrl(url), trusted }
  }
  return { base: '', trusted: true }
}

const RESOLVED_API_BASE = resolveApiBase()

/** Validated API base; the empty string means same-origin relative requests. */
export const API_BASE = RESOLVED_API_BASE.base
/** True only when API_BASE may receive the session cookie; an untrusted base must never
 *  get `credentials: 'include'` or a runtime-injected attacker base harvests the cookie. */
export const API_BASE_TRUSTED = RESOLVED_API_BASE.trusted


/** Pre-cookie key, kept only so an old build's script-readable token is deleted on sight, never read. */
const LEGACY_TOKEN_KEY = 'vccircle_auth_token'

/** Storage can throw (private mode, sandboxed frame), so the removal is guarded. */
export function clearLegacyToken(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(LEGACY_TOKEN_KEY)
  } catch {
    /* storage unavailable */
  }
}

/** Credentialed, never header-bearing: no `Authorization` header and no token in a body, so a
 *  script-readable token is one XSS bug away from a durable takeover. */
export function authRequestInit(init?: RequestInit): RequestInit {
  if (!API_BASE_TRUSTED) return init ?? {}
  return { ...init, credentials: 'include' }
}

export interface AuthUser {
  id: string
  email: string
  name: string
  role: string
  is_active: boolean
}

let meCache: AuthUser | null | undefined
let meCacheTs = 0

// 0 disables time-based expiry, leaving clearMeCache() as the only invalidation.
const ME_CACHE_TTL_RAW = Number(process.env.NEXT_PUBLIC_ME_CACHE_TTL_MS || 60000)
const ME_CACHE_TTL_MS = Number.isNaN(ME_CACHE_TTL_RAW) ? 60000 : ME_CACHE_TTL_RAW

/** Network failures are RETHROWN, never returned as null: only a 401 means logged out.
 *
 *  CACHE INVARIANT: memoised in MODULE state and invalidated ONLY by clearMeCache() — which
 *  login and a 401 both call — or the TTL, so a component must treat `prop present` and
 *  `prop undefined` as different signals. */
export async function getMe(force = false, signal?: AbortSignal | null): Promise<AuthUser | null> {
  clearLegacyToken()
  const fresh = meCache !== undefined && (ME_CACHE_TTL_MS <= 0 || Date.now() - meCacheTs < ME_CACHE_TTL_MS)
  if (!force && fresh) return meCache ?? null
  // Deadline stays armed across res.json(); clear() must run once, in finally.
  const deadline = createDeadline(ME_DEADLINE_MS, signal ?? null)
  try {
    const res = await fetch(
      `${API_BASE}/api/auth/me`,
      authRequestInit({ signal: deadline.signal })
    )
    if (res.status === 401) {
      // Definitive logged-out: invalidate the cache.
      clearMeCache()
      clearLegacyToken()
      return null
    }
    if (!res.ok) return null
    try {
      meCache = (await res.json()) as AuthUser
    } catch (err) {
      // In the deadline's scope, so this abort is a cancellation or timeout, NOT a malformed
      // payload — swallowing it would resolve null, the logged-out sentinel.
      if (deadline.timedOut() || (err as Error)?.name === 'AbortError') throw err
      // Genuine malformed 200: return null because callers may have no .catch.
      console.error('getMe: failed to parse /api/auth/me response')
      return null
    }
  } catch (err) {
    // A caller abort (unmount, or the dashboard losing its identity race) is a cancellation,
    // not a fault: logging it would report a healthy auth service as down on every teardown.
    if (!deadline.timedOut() && (err as Error)?.name !== 'AbortError') {
      console.error('getMe: failed to reach the auth service', err)
    } else if (deadline.timedOut()) {
      console.error('getMe: /api/auth/me timed out', err)
    }
    throw err
  } finally {
    deadline.clear()
  }
  meCacheTs = Date.now()
  return meCache
}

export function clearMeCache(): void {
  meCache = undefined
  meCacheTs = 0
}

/** `keepalive: true` is load-bearing: `redirectToLogin` runs `window.location.replace` on the
 *  next line, which would cancel an ordinary fetch and leave an httpOnly session live in both
 *  the browser and the store, where script cannot clear it. */
export function logout(): void {
  const deadline = createDeadline(LOGOUT_DEADLINE_MS)
  fetch(
    `${API_BASE}/api/auth/logout`,
    authRequestInit({ method: 'POST', keepalive: true, signal: deadline.signal })
  )
    .catch(() => {})
    .finally(() => deadline.clear())
  redirectToLogin()
}

export function redirectToLogin(next?: string): void {
  clearMeCache()
  if (typeof window !== 'undefined') {
    // `next` only when safe: `//evil.com` or `https://evil.com` must fall back to a plain `/login`.
    const target = isSafeRedirect(next) ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}