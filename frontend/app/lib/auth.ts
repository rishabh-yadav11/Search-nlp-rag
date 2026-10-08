'use client'
import { devApiBase, parseApiBaseUrl, readApiBaseEnv } from './api-base'
import { isSafeRedirect } from './safe-url'
import { createDeadline, LOGOUT_DEADLINE_MS, ME_DEADLINE_MS } from './deadline'

// Hosts explicitly allowed to receive the session cookie. Only matters for a
// runtime-injected `window.API_BASE`; build-time config is operator-controlled
// and trusted. Comma-separated, e.g. "api.example.com".
const TRUSTED_API_HOSTS: string[] = (process.env.NEXT_PUBLIC_TRUSTED_API_HOSTS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)

// Candidates, in priority order. `window.API_BASE` is runtime/injectable (e.g.
// XSS payload, malicious inline script) and therefore untrusted unless its host
// is allow-listed. The build-time env var and the dev default are
// operator-controlled but obey the SAME trust rule: a cross-origin base is
// trusted only over https AND allow-listed, or when it is loopback (http
// loopback is not a cleartext risk). No non-loopback cross-origin http base may
// ever be trusted. `app/lib/api-base.ts` owns the `NEXT_PUBLIC_API_BASE` read,
// so this browser bundle and the edge middleware can never disagree about what
// a valid base is.
const ENV_API_BASE = readApiBaseEnv()
const DEV_API_BASE = devApiBase()


/** True when `url` resolves to the current document's origin; an empty/relative
 *  base also resolves there, so callers must treat an empty base as same-origin. */
function isSameOrigin(url: URL): boolean {
  if (typeof window === 'undefined') return false
  return url.origin === window.location.origin
}

/** True when `url` uses the https scheme. Required for any cross-origin trusted
 *  base so session credentials are never sent over cleartext http. */
function isHttps(url: URL): boolean {
  return url.protocol === 'https:'
}

/** True when `url` is loopback (`localhost`, `127.0.0.1`, `[::1]`). Loopback
 *  traffic never leaves the machine, so http over loopback is NOT a cleartext
 *  risk — a dev backend like `http://localhost:8000` is trusted cross-origin. */
function isLoopback(url: URL): boolean {
  // `URL.hostname` returns IPv6 loopback without brackets (e.g. `::1`).
  return ['localhost', '127.0.0.1', '::1'].includes(url.hostname)
}

/** Normalize an allow-list entry or URL host: strip any leading scheme
 *  (`https://`), trailing slash, trailing dot, and a default port (`:80` /
 *  `:443`). The URL parser already drops the default port from `url.host`, so
 *  entries must too or `api.example.com:443` would never match. Non-default
 *  ports are preserved so matching stays exact `host:port`. Lowercasing is
 *  required because `url.host` is lowercased but entries are only `trim()`-ed
 *  (hostnames are case-insensitive per RFC 4343). */
export function normalizeHost(host: string): string {
  return host
    .replace(/^https?:\/\//, '')
    .replace(/\/+$/, '')
    .replace(/\.$/, '')
    .replace(/:(?:80|443)$/, '')
    .toLowerCase()
}

/** True when `url`'s host is present in NEXT_PUBLIC_TRUSTED_API_HOSTS (compared
 *  by exact `host:port`, so `host` and `host:8443` are distinct). */
function hostInAllowList(url: URL): boolean {
  const h = normalizeHost(url.host)
  return TRUSTED_API_HOSTS.some((entry) => normalizeHost(entry) === h)
}

/** Request base from a parsed base: `origin + pathname` keeps a path prefix
 *  (`https://host/v1`) that a bare origin would discard; query/fragment drop. */
function baseFromUrl(url: URL): string {
  // Strip a trailing slash: `new URL('http://host').pathname` is '/' so callers
  // that concatenate `${API_BASE}/api/...` would otherwise produce a `//api`
  // double slash the backend 404s. A base with a real sub-path keeps it.
  return url.origin + url.pathname.replace(/\/+$/, '')
}

/** Trust rule for the session cookie: first-party is trusted — empty (relative
 *  → window.location.origin), same-origin (http allowed for local dev), or
 *  loopback. An explicit cross-origin allow-list entry is trusted only over
 *  https. Every other cross-origin host is untrusted and must NOT receive
 *  credentials, so the session never crosses an unencrypted channel. */
function isTrustedBase(base: string, url: URL | null): boolean {
  if (!base) return true // empty relative base → same-origin first-party
  if (!url) return false
  if (isSameOrigin(url)) return true // same-origin (http ok for local dev)
  // Loopback (e.g. http://localhost:8000): cross-origin but no cleartext risk.
  if (isLoopback(url)) return true
  // Cross-origin: trusted only when https AND explicitly allow-listed.
  return isHttps(url) && hostInAllowList(url)
}

function resolveApiBase(): { base: string; trusted: boolean } {
  // A root-relative base is a same-origin base; it resolves only against a
  // current origin, so SSR passes `undefined` and takes the empty base.
  const currentOrigin = typeof window !== 'undefined' ? window.location.origin : undefined
  // Runtime-injected base (`window.API_BASE`): the only attacker-reachable
  // vector, so it is trusted only when same-origin or allow-listed, and is read
  // live (not captured at import time) so a late override is still honored.
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
  // Operator-controlled build-time config / dev default: the SAME trust rule
  // applies, so a cross-origin http base is never trusted.
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
  // No base set → same-origin relative requests, which are first-party.
  return { base: '', trusted: true }
}

const RESOLVED_API_BASE = resolveApiBase()

/** The validated API base. Empty string means same-origin relative requests. */
export const API_BASE = RESOLVED_API_BASE.base
/** True only when API_BASE is a trusted backend allowed to receive the session.
 *  The session is an httpOnly cookie the browser attaches to any
 *  `credentials: 'include'` request, so this flag decides whether we use that
 *  mode at all; `authRequestInit` omits `credentials` when it is false. */
export const API_BASE_TRUSTED = RESOLVED_API_BASE.trusted


/** Pre-cookie localStorage key, kept ONLY to delete a token written by an older
 *  build. A leftover copy is the XSS-exfiltratable credential the httpOnly
 *  cookie replaced, so it is deleted on sight rather than read. */
const LEGACY_TOKEN_KEY = 'vccircle_auth_token'

/** Delete the legacy `vccircle_auth_token` localStorage entry, if present.
 *  Storage can throw (private mode, sandboxed iframe), so access is guarded. */
export function clearLegacyToken(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(LEGACY_TOKEN_KEY)
  } catch {
    /* storage unavailable */
  }
}

/** RequestInit for a session-bearing API call. SECURITY:
 *  `credentials: 'include'` attaches the httpOnly cookie, so it is set ONLY
 *  for a trusted API_BASE; an untrusted base gets the init UNCHANGED,
 *  deliberately with no credentials so the request stays unauthenticated.
 *  There is no `Authorization` header: the backend no longer accepts bearer
 *  tokens and JS cannot read the cookie at all. */
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

// TTL so role/is_active changes are eventually picked up; 0 disables expiry.
const ME_CACHE_TTL_RAW = Number(process.env.NEXT_PUBLIC_ME_CACHE_TTL_MS || 60000)
const ME_CACHE_TTL_MS = Number.isNaN(ME_CACHE_TTL_RAW) ? 60000 : ME_CACHE_TTL_RAW

/**
 * Fetch the current authenticated user (`/api/auth/me`) via the httpOnly session
 * cookie. null when the session is missing/rejected (401) or on any non-2xx; a
 * network/transport failure RE-THROWS with the cookie untouched, so callers can
 * tell a transient failure from a definitive logout and MUST NOT treat it as
 * one. Never redirects.
 *
 * The only "am I logged in?" predicate: JS cannot read the cookie, so there is
 * no synchronous stored flag to consult.
 *
 * `signal` lets the analytics dashboard actually cancel the socket when it
 * abandons its own race; such an abort leaves `deadline.timedOut()` false, so it
 * is never reported as a backend timeout.
 *
 * CACHE INVARIANT: a cached record belongs to whichever session was live when
 * it was stored, and the only events that swap the session under a live page are
 * a login and a 401/logout — both call `clearMeCache()`. That is what keeps
 * user A's record from being served to user B after B signs in on the same tab.
 */
export async function getMe(force = false, signal?: AbortSignal | null): Promise<AuthUser | null> {
  // A pre-cookie build may have left a readable token in localStorage.
  clearLegacyToken()
  const fresh = meCache !== undefined && (ME_CACHE_TTL_MS <= 0 || Date.now() - meCacheTs < ME_CACHE_TTL_MS)
  if (!force && fresh) return meCache ?? null
  // Deadline so a hung auth service cancels the request instead of leaking one
  // in-flight `/api/auth/me` per poll tick. It must stay armed across
  // `res.json()` too, not just the header read — a body that never completes
  // would otherwise hang forever with the timer already disarmed — so the whole
  // exchange lives inside the try and `clear()` happens once, in `finally`.
  const deadline = createDeadline(ME_DEADLINE_MS, signal ?? null)
  try {
    // Credentialed, not header-bearing: the browser attaches the httpOnly cookie
    // and JS cannot read it. There is no `Authorization` header any more.
    const res = await fetch(
      `${API_BASE}/api/auth/me`,
      authRequestInit({ signal: deadline.signal })
    )
    if (res.status === 401) {
      // Definitive "logged out": drop the cached user and any legacy token.
      clearMeCache()
      clearLegacyToken()
      return null
    }
    if (!res.ok) return null
    try {
      meCache = (await res.json()) as AuthUser
    } catch (err) {
      // The body read is inside the deadline's scope, so an abort here is a
      // cancellation or a timeout, NOT a malformed payload. Swallowing it would
      // resolve `null` — the "definitive logged out" sentinel — so a hung auth
      // service would masquerade as a rejected session.
      if (deadline.timedOut() || (err as Error)?.name === 'AbortError') throw err
      // Genuine malformed/non-JSON 200: return null safely rather than throw.
      console.error('getMe: failed to parse /api/auth/me response')
      return null
    }
  } catch (err) {
    // Network/transport failure: do NOT treat as "not authenticated" (the cookie
    // is left intact for a later retry) and rethrow rather than return null, so
    // callers can tell this from a definitive 401. A caller-initiated abort
    // (page teardown, abandoned race) is a cancellation, not a backend fault, so
    // it is not logged as one.
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

/**
 * Sign out: revoke the session server-side and send the user to `/login`.
 *
 * Credentialed, not header-bearing (httpOnly cookie, no `Authorization` header),
 * and fire-and-forget: the page moves on whether or not the POST succeeds, so a
 * network failure cannot strand the user on an authenticated page.
 * `redirectToLogin` already clears the cached identity and any legacy token.
 *
 * `keepalive: true` is load-bearing, not decoration. `redirectToLogin` calls
 * `window.location.replace`, which tears the page down and cancels an ordinary
 * fetch before the revocation reaches the server; script cannot clear an
 * httpOnly cookie at all, so a lost POST leaves the session live in the browser
 * AND in the store. `keepalive` lets the request outlive the navigation; the
 * deadline only releases the socket and never inside that window. Failures are
 * swallowed either way.
 */
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

/** Redirect to `/login`; `next` is preserved only when it is a safe path. */
export function redirectToLogin(next?: string): void {
  clearMeCache()
  if (typeof window !== 'undefined') {
    // Only preserve `next` when safe — `//evil.com` must fall back to `/login`.
    const target = isSafeRedirect(next) ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}