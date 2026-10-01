/** Plain by necessity: the edge runtime imports this and cannot pull in React, a `'use client'` module, or read `window` at import time. */

import { CONTROL_CHAR_RE } from './safe-url'

export const DEV_LOOPBACK = 'http://localhost:8001'

/**
 * Next inlines only a literal `process.env.NEXT_PUBLIC_API_BASE` expression and the client
 * bundle has no `process` polyfill, so this read must stay spelled out; a test fails on a second reader.
 */
export function readApiBaseEnv(): string {
  return process.env.NEXT_PUBLIC_API_BASE || ''
}

export function devApiBase(): string {
  return process.env.NODE_ENV === 'development' ? DEV_LOOPBACK : ''
}

/**
 * `;` and whitespace would inject a second CSP `connect-src` directive, control chars must be
 * checked raw because the URL parser silently strips tab/CR/LF, and a base carrying credentials
 * would replay them on every request.
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
 * `connect-src` is an origin list, so the path is dropped; an unusable value yields `''` rather
 * than an operator-influenced source.
 */
export function sanitizeApiBaseOrigin(value: unknown): string {
  const url = parseApiBaseUrl(value)
  return url ? url.origin : ''
}
