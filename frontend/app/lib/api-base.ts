/**
 * The single place the frontend reads, validates and projects an API base.
 *
 * Deliberately plain — no React, no `'use client'`, no Node builtins, no
 * `window` at import time — so the edge runtime (`frontend/middleware.ts`,
 * which cannot import a `'use client'` module) and the browser bundle
 * (`app/lib/auth.ts`) share one validator, one env read, one dev constant.
 */

import { CONTROL_CHAR_RE } from './safe-url'

/** Dev backend loopback: the `connect-src` fallback and dev request base when no base is configured. */
export const DEV_LOOPBACK = 'http://localhost:8001'

/**
 * The configured API base, or `''` when unset.
 *
 * `process.env.NEXT_PUBLIC_API_BASE` MUST be written out literally here: Next
 * substitutes that exact member expression at build time, so reading the value
 * any other way is `undefined` in the browser bundle while typecheck, tests
 * and the build stay green. This is also the only module that READS it; a
 * test scans the source to keep it that way.
 */
export function readApiBaseEnv(): string {
  return process.env.NEXT_PUBLIC_API_BASE || ''
}

/** The loopback origin in development, `''` elsewhere (same-origin). */
export function devApiBase(): string {
  return process.env.NODE_ENV === 'development' ? DEV_LOOPBACK : ''
}

/**
 * Validate a candidate API base and return it parsed, or `null`.
 *
 * Rejects non-strings and empty values; whitespace or `;` (host confusion, and
 * `;` would inject a `connect-src` directive); C0 controls and DEL (the URL
 * parser silently drops tab/CR/LF — the reason `safe-url.ts` refuses them for
 * navigation targets too, which is why that check is imported, not re-written);
 * a root-relative base with no `base` origin to resolve against (the edge
 * runtime has none); any non-`http(s)` scheme; and embedded credentials.
 *
 * The frontend's only base validator: a *configuration* check, distinct from
 * `safe-url.ts`'s navigation guard because a base may carry a path.
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
 * A bare http(s) `origin` suitable for a `connect-src` source, or `''` when
 * the value is not a usable base. The path is dropped: `connect-src` is an
 * origin list. A projection of `parseApiBaseUrl`, not a second validation.
 */
export function sanitizeApiBaseOrigin(value: unknown): string {
  const url = parseApiBaseUrl(value)
  return url ? url.origin : ''
}
