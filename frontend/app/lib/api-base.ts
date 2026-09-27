/**
 * The single place the frontend reads, validates and projects an API base.
 *
 * This module is deliberately plain: no React, no `'use client'`, no Node
 * builtins, and no `window` access at import time. That is what lets the edge
 * runtime (`frontend/middleware.ts`, which cannot import a `'use client'`
 * module) and the browser bundle (`app/lib/auth.ts`) share ONE validator, ONE
 * `NEXT_PUBLIC_API_BASE` read and ONE dev-loopback constant. Before this
 * module existed each side had its own copy of all three, and they disagreed
 * in both directions: the middleware's copy accepted `https://user:pw@host`,
 * which `auth.ts` rejected, while rejecting a `;` that `auth.ts` accepted.
 *
 * The only remaining difference between consumers is what they do with a parsed
 * URL, and that difference is intentional and documented at each call site:
 * the CSP wants a bare `origin` (a `connect-src` source), while the client
 * also wants the base `pathname` (`https://host/v1`). Two projections of one
 * validation, not two validators.
 */

import { CONTROL_CHAR_RE } from './safe-url'

/**
 * The dev backend loopback origin. Used as a `connect-src` fallback when
 * `NEXT_PUBLIC_API_BASE` is unset (so `next dev` on :3000 can reach the API on
 * :8001 without the request being CSP-blocked) and as the dev default request
 * base. A constant, so it is always safe to emit.
 */
export const DEV_LOOPBACK = 'http://localhost:8001'

/**
 * The configured API base, or `''` when unset.
 *
 * The `process.env.NEXT_PUBLIC_API_BASE` member expression below MUST be
 * written out literally, here, for a build-time reason that is easy to undo by
 * accident: Next inlines a `NEXT_PUBLIC_*` value by substituting the exact
 * member expression `process.env.NEXT_PUBLIC_API_BASE` (see `getDefineEnv` in
 * next/dist/build/define-env.js, which defines exactly those keys, and notes
 * that the client bundle has no `process` polyfill to fall back on). Reading
 * the same value as a property of an env object handed in from a caller — the
 * injectable shape that reads better in a test — is NOT substituted, so the
 * browser bundle would silently see `undefined` and fall back to same-origin
 * relative requests while typecheck, tests and the build all stayed green.
 *
 * This is also the only place in the frontend that READS the variable (other
 * modules name it in comments and in the operator-facing warning string);
 * a test scans the source to keep it that way.
 */
export function readApiBaseEnv(): string {
  return process.env.NEXT_PUBLIC_API_BASE || ''
}

/**
 * The dev-only request base: the loopback origin in development, `''`
 * everywhere else. Production with no configured base falls back to
 * same-origin relative requests.
 *
 * Single-sources the "is this a development environment" decision — which both
 * `app/lib/auth.ts` (request base) and `frontend/middleware.ts` (`connect-src`
 * fallback) previously had to make for themselves — and returns the one
 * `DEV_LOOPBACK` constant above rather than a second literal.
 */
export function devApiBase(): string {
  return process.env.NODE_ENV === 'development' ? DEV_LOOPBACK : ''
}

/**
 * Validate a candidate API base and return it parsed, or `null` when it is not
 * usable. Rejects:
 *  - anything that is not a non-empty string,
 *  - whitespace or `;` anywhere (host confusion, and a `;` would inject an
 *    extra CSP directive into a `connect-src` source),
 *  - C0 controls and DEL (the URL parser silently REMOVES tab/CR/LF, so a value
 *    carrying one can parse to something the raw string never said — the same
 *    reason `app/lib/safe-url.ts` refuses them for navigation targets, which
 *    is why the check is imported rather than re-written),
 *  - a root-relative base unless a `base` origin is supplied to resolve it
 *    against; the edge runtime has no current origin, so it refuses one,
 *  - any non-`http(s)` scheme (`javascript:`, `data:`, …),
 *  - embedded credentials (`https://user:pw@host`).
 *
 * This is the only base validator in the frontend. It is a *configuration*
 * validator and is intentionally distinct from the navigation guard in
 * `app/lib/safe-url.ts`: a base may carry a path, and a relative base is
 * meaningless without an origin to resolve it against.
 */
export function parseApiBaseUrl(value: unknown, base?: string): URL | null {
  if (typeof value !== 'string' || value.length === 0) return null
  if (/[\s;]/.test(value)) return null
  if (CONTROL_CHAR_RE.test(value)) return null
  if (value.startsWith('/')) {
    if (!base) return null
    try {
      return new URL(value, base)
    } catch {
      return null
    }
  }
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return null
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null
  if (url.username || url.password) return null
  return url
}

/**
 * A bare http(s) `origin` suitable for a `connect-src` source, or `''` when the
 * value is not a usable base. Falls back to `''` rather than emitting an
 * attacker- or operator-influenced value into a CSP directive. The path is
 * dropped on purpose: `connect-src` is an origin list. This is a projection of
 * `parseApiBaseUrl`, not a second validation of the same input.
 */
export function sanitizeApiBaseOrigin(value: unknown): string {
  const url = parseApiBaseUrl(value)
  return url ? url.origin : ''
}
