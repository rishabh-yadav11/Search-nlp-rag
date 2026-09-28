'use client'
import { devApiBase, parseApiBaseUrl, readApiBaseEnv } from './api-base'
import { isSafeRedirect } from './safe-url'
import { createDeadline, LOGOUT_DEADLINE_MS, ME_DEADLINE_MS } from './deadline'

// Hosts explicitly allowed to receive the session cookie. This only matters
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
// unless its host is explicitly allow-listed. The build-time env var and the dev
// default are operator-controlled, but they still obey the SAME trust rule:
// a cross-origin base is trusted only over https AND when allow-listed, or when
// it is a loopback address (http loopback is not a network cleartext risk). No
// non-loopback cross-origin http base may ever be trusted.
// The configured base and the dev default come from `app/lib/api-base.ts`, the
// one module that owns the `NEXT_PUBLIC_API_BASE` read, the validation and the
// dev-loopback constant, so this browser bundle and the edge middleware can
// never disagree about what a valid base is.
const ENV_API_BASE = readApiBaseEnv()
const DEV_API_BASE = devApiBase()


/** True when `url` resolves to the same origin as the current document. An
 *  empty/relative base also resolves to same-origin, so callers should treat an
 *  empty base as same-origin too (see resolveApiBase). */
function isSameOrigin(url: URL): boolean {
  if (typeof window === 'undefined') return false
  return url.origin === window.location.origin
}

/** True when `url` uses the https scheme. Required for any cross-origin trusted
 *  base so session credentials are never sent over cleartext http. */
function isHttps(url: URL): boolean {
  return url.protocol === 'https:'
}

/** True when `url` resolves to a loopback address (`localhost`, `127.0.0.1`,
 *  `[::1]`). Loopback traffic never leaves the machine, so http over loopback is
 *  NOT a network cleartext risk — a dev backend like `http://localhost:8000` is
 *  therefore trusted even though it is cross-origin http. */
function isLoopback(url: URL): boolean {
  // `URL.hostname` returns IPv6 loopback without brackets (e.g. `::1`).
  return ['localhost', '127.0.0.1', '::1'].includes(url.hostname)
}

/** Normalize an allow-list entry or URL host for comparison: strip any leading
 *  scheme (`https://` / `http://`), any trailing slash, any trailing dot, and a
 *  trailing default port (`:80` for http, `:443` for https). A default port is
 *  not a distinguishing feature — the URL parser already drops it from
 *  `url.host` (`URL('https://host:443').host === 'host'`) — so it MUST be
 *  dropped from allow-list entries too, or an entry like `api.example.com:443`
 *  would never match. Explicit non-default ports (e.g. `:8080`) are preserved,
 *  so matching stays exact `host:port` (`api.example.com:8080` matches only
 *  `api.example.com:8080`, not `api.example.com`). Hostnames are also
 *  lowercased: `url.host` is always lowercased, but allow-list entries are
 *  only `trim()`-ed, so an entry like `API.Example.COM` would otherwise
 *  silently fail to match (hostnames are case-insensitive per RFC 4343). */
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

/** Resolve the request base URL from a parsed API base, preserving any path
 *  (e.g. `https://host/v1` → `https://host/v1`) rather than discarding it to
 *  just the origin. Query/fragment are dropped as they are not part of a base. */
function baseFromUrl(url: URL): string {
  return url.origin + url.pathname
}

/** Trust rule: a base is trusted (allowed to receive the session cookie) when
 *  it is first-party — i.e. empty (relative → window.location.origin), same-origin
 *  (http allowed for local dev), or a loopback address (e.g. http://localhost:
 *  8000; cross-origin but never a network cleartext risk). An explicit
 *  cross-origin host in NEXT_PUBLIC_TRUSTED_API_HOSTS is trusted only over
 *  https. A non-loopback cross-origin http base is NEVER trusted so session
 *  credentials are never sent over an unencrypted channel. Any other
 *  cross-origin host is untrusted and must NOT receive credentials. */
function isTrustedBase(base: string, url: URL | null): boolean {
  if (!base) return true // empty relative base → same-origin first-party
  if (!url) return false
  if (isSameOrigin(url)) return true // same-origin (http ok for local dev)
  // Loopback (e.g. http://localhost:8000) is cross-origin but never a network
  // cleartext risk, so it is trusted even over http.
  if (isLoopback(url)) return true
  // Cross-origin: only trusted when https AND explicitly allow-listed. A
  // non-loopback cross-origin http base is NEVER trusted (cleartext).
  return isHttps(url) && hostInAllowList(url)
}

function resolveApiBase(): { base: string; trusted: boolean } {
  // A root-relative base is a same-origin base. It can only be resolved when
  // there is a current origin, so SSR passes `undefined` and falls back to the
  // safe empty (same-origin) base.
  const currentOrigin = typeof window !== 'undefined' ? window.location.origin : undefined
  // Runtime-injected base (`window.API_BASE`): it is the only attacker-reachable
  // vector (XSS / malicious inline script), so it is trusted ONLY when it is a
  // valid http(s) URL that is same-origin OR explicitly allow-listed. A
  // cross-origin override REQUIRES NEXT_PUBLIC_TRUSTED_API_HOSTS to be set at
  // build; otherwise it is ignored and we fall back to the safe same-origin
  // base (which is itself trusted first-party). `window.API_BASE` is read live
  // here (not captured at import time) so an override set after module load is
  // still honored.
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
  // Operator-controlled build-time config / dev default: apply the SAME trust
  // rule as any other source. A cross-origin base is trusted ONLY over https
  // AND when explicitly allow-listed; a cross-origin http base (e.g. an
  // operator setting `NEXT_PUBLIC_API_BASE=http://host`) must NEVER be trusted,
  // so session credentials are not sent over cleartext.
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
  // Production: no base set → same-origin relative requests. This is
  // first-party, so it IS trusted and the token is attached. NEXT_PUBLIC_API_BASE
  // must be set at build for a real backend.
  return { base: '', trusted: true }
}

const RESOLVED_API_BASE = resolveApiBase()

/** The validated API base. Empty string means same-origin relative requests. */
export const API_BASE = RESOLVED_API_BASE.base
/** True only when API_BASE is a trusted backend allowed to receive the session.
 *  Since the session now lives in an httpOnly cookie, the browser attaches it
 *  to any request made with `credentials: 'include'` — this flag is what
 *  decides whether we set that mode at all. An untrusted base therefore MUST
 *  NEVER receive credentials: `authRequestInit` omits `credentials` there so a
 *  runtime-injected attacker base cannot harvest the session cookie. */
export const API_BASE_TRUSTED = RESOLVED_API_BASE.trusted


/** The pre-cookie localStorage key, kept ONLY to delete any token written by an
 *  older build. The session is an httpOnly cookie; a live localStorage copy of
 *  it is exactly the XSS-exfiltratable credential this design removed, so any
 *  leftover value is deleted on sight rather than read. */
const LEGACY_TOKEN_KEY = 'vccircle_auth_token'

/** Delete the legacy `vccircle_auth_token` localStorage entry, if present.
 *  Storage can throw (private mode, disabled cookies, sandboxed iframe), so
 *  every access is guarded. Called from getMe() and after a successful login so
 *  a pre-cookie session token cannot linger and be re-read. */
export function clearLegacyToken(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(LEGACY_TOKEN_KEY)
  } catch {
    /* storage unavailable */
  }
}

/** Build the RequestInit for a session-bearing API call.
 *  SECURITY: `credentials: 'include'` makes the browser attach the httpOnly
 *  session cookie, so it is set ONLY when API_BASE is a trusted backend
 *  (same-origin or an explicitly allow-listed host). Sending it to an
 *  attacker-controlled/untrusted base would leak the session cross-origin.
 *  See API_BASE_TRUSTED / resolveApiBase above. When the base is untrusted the
 *  init is returned UNCHANGED — deliberately no `credentials`, so the request
 *  stays unauthenticated rather than carrying the cookie to a foreign host.
 *  There is deliberately no `Authorization` header any more: the backend no
 *  longer accepts bearer tokens, and JS cannot read the cookie at all. */
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

// Configurable TTL so role/is_active changes are eventually picked up even if
// the caller forgets to clear the cache. 0 disables the time-based expiry.
const ME_CACHE_TTL_RAW = Number(process.env.NEXT_PUBLIC_ME_CACHE_TTL_MS || 60000)
const ME_CACHE_TTL_MS = Number.isNaN(ME_CACHE_TTL_RAW) ? 60000 : ME_CACHE_TTL_RAW

/** Fetch the current authenticated user (`/api/auth/me`) by sending the
 *  httpOnly session cookie. Returns null when the session is missing/rejected
 *  (401) or on a non-2xx response. On a network/transport failure the session
 *  is left alone (the cookie is untouched) so a later retry can recover — the
 *  fetch error is rethrown so callers can distinguish a transient network
 *  failure from a definitive "logged out" (null) and MUST NOT treat it as a
 *  logout. Never redirects.
 *
 *  This is the ONLY "am I logged in?" predicate: JS cannot read the cookie, so
 *  there is no synchronous stored flag to consult.
 *
 *  `signal` is an optional caller signal — the analytics dashboard passes its
 *  load controller so abandoning the dashboard's own 10 s race really does
 *  cancel the socket. Aborting it leaves `deadline.timedOut()` false, so it is
 *  never reported as a backend timeout.
 *
 *  CACHE INVARIANT: a cached record belongs to whichever session was live when
 *  it was stored, and the only events that can swap the session under a live
 *  page are a login and a 401/logout. So the cache is keyed on TIME alone, and
 *  every identity-changing event — login success and 401 — calls
 *  clearMeCache(). That is what guarantees user A's record is never served to
 *  user B after B signs in on the same tab. */
export async function getMe(force = false, signal?: AbortSignal | null): Promise<AuthUser | null> {
  // A pre-cookie build may have left a readable session token behind; delete it
  // rather than let it sit in localStorage exfiltratable.
  clearLegacyToken()
  const fresh = meCache !== undefined && (ME_CACHE_TTL_MS <= 0 || Date.now() - meCacheTs < ME_CACHE_TTL_MS)
  if (!force && fresh) return meCache ?? null
  // Deadline so a hung auth service actually cancels the request instead of
  // leaking one in-flight `/api/auth/me` per poll tick. The dashboard races
  // this with its own 10 s guard and abandons the promise, which never
  // cancelled the underlying fetch; aborting here is what makes that safe.
  //
  // The deadline must stay armed across `res.json()` too, not just the header
  // read: a response whose headers arrive but whose body never completes
  // would otherwise hang forever with the timer already disarmed. So the whole
  // exchange lives inside the try, and `clear()` happens once, in `finally`.
  const deadline = createDeadline(ME_DEADLINE_MS, signal ?? null)
  try {
    // The session is an httpOnly cookie the browser attaches for us and JS
    // cannot read, so this call is credentialed rather than header-bearing:
    // `authRequestInit` attaches the cookie when API_BASE is a trusted backend
    // and omits `credentials` when it is not. There is no `Authorization`
    // header any more.
    const res = await fetch(
      `${API_BASE}/api/auth/me`,
      authRequestInit({ signal: deadline.signal })
    )
    if (res.status === 401) {
      // Definitive "logged out": drop the cached user so neither a stale record
      // nor a previous session's legacy token survives the rejection.
      clearMeCache()
      clearLegacyToken()
      return null
    }
    if (!res.ok) return null
    try {
      meCache = (await res.json()) as AuthUser
    } catch (err) {
      // The body read is inside the deadline's scope, so an abort here is a
      // cancellation or a timeout, NOT a malformed payload. Swallowing it
      // would resolve `null` — this function's "definitive logged out"
      // sentinel — and the outer classification below would never run, so a
      // hung auth service would masquerade as a rejected session. This is the
      // same guard `api()` in chat/page.tsx and the For You page apply to
      // their own reads.
      if (deadline.timedOut() || (err as Error)?.name === 'AbortError') throw err
      // Genuine malformed/non-JSON 200: don't throw (callers may lack a
      // .catch); treat as an unexpected payload and return null safely.
      console.error('getMe: failed to parse /api/auth/me response')
      return null
    }
  } catch (err) {
    // Network/transport failure: do NOT treat as "not authenticated" (the
    // session cookie is left intact so a later retry can succeed). Rethrow
    // rather than return null so callers can tell this apart from a definitive
    // 401/logged-out null.
    //
    // A caller-initiated abort (the dashboard unmounting, or giving up on its
    // 10 s identity race) is a cancellation, not a backend fault. Logging it as
    // "failed to reach the auth service" would report a perfectly healthy auth
    // service as down on every page teardown, so only real transport failures
    // and real timeouts are logged.
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
 * Sign out: revoke the session server-side and send the user to `/login`. The
 * single implementation — the search page, the chat sidebar and the analytics
 * dashboard all call this instead of each open-coding the same POST plus
 * redirect.
 *
 * The request is CREDENTIALED, not header-bearing: the session is an httpOnly
 * cookie that the browser attaches for us and JS cannot read, so there is no
 * `Authorization` header to set. `authRequestInit` attaches the cookie only
 * when API_BASE is a trusted backend. `redirectToLogin` already clears the
 * cached identity and any legacy localStorage token, so neither is repeated
 * here. The logout request is fire-and-forget on purpose: the page moves on
 * whether or not the backend call succeeds, so a network failure cannot leave
 * the user stuck on an authenticated page.
 *
 * `keepalive: true` is load-bearing, not decoration. `redirectToLogin` calls
 * `window.location.replace`, which tears the page down; an ordinary fetch is
 * cancelled with it and the revocation never reaches the server. That used to
 * be survivable: the credential lived in localStorage and `clearToken()` had
 * already destroyed it synchronously, so a lost POST cost nothing. An httpOnly
 * cookie cannot be cleared by script at all, so a lost POST means the session
 * stays live in the browser AND in the store — the user lands on /login still
 * authenticated, and a shared machine stays signed in. `keepalive` lets the
 * request outlive the navigation.
 *
 * Bounded like every other request here (#287): a backend that accepts the
 * connection and never answers would otherwise hold the socket open forever.
 * The bound only ever releases that socket, and only at 10 s — long after the
 * redirect that `keepalive` has to outlive — so it never costs the revocation
 * its window. A failure is swallowed either way.
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

/** Redirect to the login page (used when the backend rejects the session).
 *  `next` (a path) is preserved so the user is sent back after signing in. */
export function redirectToLogin(next?: string): void {
  clearMeCache()
  if (typeof window !== 'undefined') {
    // Only preserve `next` when it is a safe, root-relative path. A value like
    // `//evil.com` or `https://evil.com` must fall back to a plain `/login`.
    const target = isSafeRedirect(next) ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}