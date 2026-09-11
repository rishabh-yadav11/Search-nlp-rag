import { NextRequest, NextResponse } from 'next/server'

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

// Sanitize the connect-src API base to a bare http(s) origin. Rejects anything
// that could inject an extra CSP source (`'self' https://evil.com`) or an extra
// directive (a `;`) or that uses a non-http(s) scheme — falling back to '' (no
// extra connect-src source) rather than emitting an attacker-influenced value.
function sanitizeApiBase(apiBase: string): string {
  if (!apiBase || /[\s;]/.test(apiBase)) return ''
  let url: URL
  try {
    url = new URL(apiBase)
  } catch {
    return ''
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return ''
  return url.origin
}

// The dev backend loopback origin, used as a connect-src fallback when
// NEXT_PUBLIC_API_BASE is unset (so `next dev` on :3000 can reach the API on
// :8001 without the request being CSP-blocked). A constant, so it is safe to
// emit directly; still routed through sanitizeApiBase for a bare origin.
const DEV_LOOPBACK = 'http://localhost:8001'

export function middleware(request: NextRequest) {
  const nonce = crypto.randomUUID()
  const apiBase = sanitizeApiBase(process.env.NEXT_PUBLIC_API_BASE ?? '')
  // Emit the dev loopback origin only in development and only when no API base
  // is configured, so production CSP stays locked to 'self' (or the configured
  // origin) and never whitelists an extra loopback endpoint.
  const devLoopback =
    !apiBase && process.env.NODE_ENV === 'development'
      ? sanitizeApiBase(DEV_LOOPBACK)
      : ''

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
