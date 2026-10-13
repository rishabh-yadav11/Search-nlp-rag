

import { CONTROL_CHAR_RE } from './safe-url'


export const DEV_LOOPBACK = 'http://localhost:8001'


export function readApiBaseEnv(): string {
  return process.env.NEXT_PUBLIC_API_BASE || ''
}


export function devApiBase(): string {
  return process.env.NODE_ENV === 'development' ? DEV_LOOPBACK : ''
}


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


export function sanitizeApiBaseOrigin(value: unknown): string {
  const url = parseApiBaseUrl(value)
  return url ? url.origin : ''
}
