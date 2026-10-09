import type { ReactNode } from 'react'
import { redirect } from 'next/navigation'
import { cookies } from 'next/headers'
import { devApiBase, parseApiBaseUrl, readApiBaseEnv } from '../lib/api-base'

export const dynamic = 'force-dynamic'

// Server-side defense-in-depth guard for the admin analytics dashboard. The
// client dashboard ALSO enforces `role === 'admin'` and the backend endpoints
// are permission-gated, but a non-admin should never even reach the dashboard
// component, so this layout redirects first. middleware.ts is intentionally
// untouched — this guard is scoped to /analytics only.
//
// The dashboard keeps its own role check (defense in depth), so a backend blip
// here fails OPEN to the client: redirecting on a network error would bounce an
// admin to /login during an outage. Non-admin / signed-out (role absent on a
// definitive response) redirects to /login?next=… .

/** Resolve the API base exactly like the browser bundle (app/lib/auth.ts):
 *  configured env over dev loopback, parsed and projected to origin+path,
 *  trailing slashes stripped so `${base}/api/...` never becomes `//api`. This
 *  module is deliberately plain (no `window`, no `'use client'`), so the same
 *  helpers already shared with middleware work here. */
function resolveApiBase(): string {
  const source = readApiBaseEnv() || devApiBase()
  if (!source) return ''
  const url = parseApiBaseUrl(source)
  if (!url) return ''
  return url.origin + url.pathname.replace(/\/+$/, '')
}

const ME_TIMEOUT_MS = 8000

export default async function AnalyticsLayout({ children }: { children: ReactNode }) {
  const base = resolveApiBase()
  let role: string | null = null

  try {
    const cookieStore = await cookies()
    const res = await fetch(`${base}/api/auth/me`, {
      headers: { cookie: cookieStore.toString() },
      // Never cache: auth state changes are invisible to a cache and a stale
      // "admin" verdict could outlive the session that issued it.
      cache: 'no-store',
      signal: AbortSignal.timeout(ME_TIMEOUT_MS),
    })
    if (res.ok) {
      const body = (await res.json()) as { role?: unknown }
      role = typeof body.role === 'string' ? body.role : null
    }
  } catch (err) {
    // Network/timeout blip: fail open to the client dashboard, which re-checks
    // and surfaces its own "Analytics unavailable" rather than an admin being
    // bounced to /login. A definitive non-admin below still redirects.
    console.error('[analytics] server guard could not reach /api/auth/me', err)
  }

  if (role !== 'admin') {
    // `redirect()` (next/navigation) is the typing-correct redirect for a Server
    // Component: it throws NEXT_REDIRECT, which Next turns into a 307 Location
    // header. A layout cannot return a NextResponse — its return type is
    // ReactNode, so NextResponse.redirect is only valid in middleware/route
    // handlers, not here.
    redirect(`/login?next=${encodeURIComponent('/analytics/dashboard')}`)
  }

  return <>{children}</>
}
