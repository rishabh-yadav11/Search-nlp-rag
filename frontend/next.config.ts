import type { NextConfig } from 'next'

const securityHeaders = [
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
]

// Loopback backend the Next server proxies /api to. Mirrors nginx's /api
// boundary (public traffic -> nginx -> API_PORT) for SERVER-COMPONENT fetches,
// which never pass through nginx. The dashboard's server guard needs this:
// same-origin prod has no NEXT_PUBLIC_API_BASE, and a relative /api URL from a
// server component resolves to the Next server itself, which has no /api route.
// Default follows setup.sh / ecosystem.config.js (8001); override via API_PORT
// at build time.
const API_PORT = process.env.API_PORT || '8001'

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: '/:path*', headers: securityHeaders }]
  },
  async rewrites() {
    return [
      {
        // The proxy target for server-side /api calls only: nginx still fronts
        // the browser, and this Next server is loopback-bound, so no public
        // request ever reaches this rewrite.
        source: '/api/:path*',
        destination: `http://127.0.0.1:${API_PORT}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
