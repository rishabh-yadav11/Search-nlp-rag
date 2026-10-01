import { NextRequest, NextResponse } from 'next/server'
import { devApiBase, readApiBaseEnv, sanitizeApiBaseOrigin } from './app/lib/api-base'

// The base still goes through `sanitizeApiBaseOrigin` because a `connect-src` source is an origin list: never emit an operator- or attacker-influenced value into a CSP directive.

const buildCsp = (nonce: string, apiBase: string, devLoopback: string) => {
  const connectSrc = ["'self'", apiBase, devLoopback].filter(Boolean).join(' ')
  return [
    "default-src 'self'",
    // No 'unsafe-eval': dev-mode React Fast Refresh evaluates a string, which this nonce-only script-src blocks.
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
  // Emit the dev loopback origin only in development and only with no API base configured, so production CSP stays locked to 'self' or the configured origin.
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
