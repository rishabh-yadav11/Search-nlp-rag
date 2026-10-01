import { NextRequest, NextResponse } from 'next/server'
import { devApiBase, readApiBaseEnv, sanitizeApiBaseOrigin } from './app/lib/api-base'

// The connect-src API base, the dev-loopback fallback and the validity rule all
// come from `app/lib/api-base.ts` — the same module `app/lib/auth.ts` resolves
// its request base from. The base still goes through `sanitizeApiBaseOrigin`,
// which returns a bare origin and `''` for anything unusable, because a
// `connect-src` source is an origin list: never emit an operator- or
// attacker-influenced value into a CSP directive.

const buildCsp = (nonce: string, apiBase: string, devLoopback: string) => {
  const connectSrc = ["'self'", apiBase, devLoopback].filter(Boolean).join(' ')
  return [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}'`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    `connect-src ${connectSrc}`,
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
  ].join('; ')
}

export function middleware(request: NextRequest) {
  const nonce = crypto.randomUUID()
  const apiBase = sanitizeApiBaseOrigin(readApiBaseEnv())
  // Emit the dev loopback origin only in development and only when no API base is
  // configured, so production CSP never whitelists an extra loopback endpoint.
  const devLoopback = !apiBase ? sanitizeApiBaseOrigin(devApiBase()) : ''

  const csp = buildCsp(nonce, apiBase, devLoopback)

  const requestHeaders = new Headers(request.headers)
  requestHeaders.set('x-csp-nonce', nonce)
  requestHeaders.set('Content-Security-Policy', csp)

  const response = NextResponse.next({ request: { headers: requestHeaders } })
  response.headers.set('Content-Security-Policy', csp)
  response.headers.set('x-csp-nonce', nonce)
  return response
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|.*\\.(?:js|css|png|jpg|svg|ico|webp)$).*)'],
}
