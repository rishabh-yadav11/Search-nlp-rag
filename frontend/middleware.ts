import { NextRequest, NextResponse } from 'next/server'

const buildCsp = (nonce: string, apiBase: string) => {
  const connectSrc = ["'self'", apiBase].filter(Boolean).join(' ')
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
  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? ''

  const csp = buildCsp(nonce, apiBase)

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
