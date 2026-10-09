import type { ReactNode } from 'react'
import { redirect } from 'next/navigation'
import { cookies } from 'next/headers'
import { readApiBaseEnv, parseApiBaseUrl } from '../lib/api-base'

export const dynamic = 'force-dynamic'

// Server-side guard for the admin analytics dashboard. The client dashboard
// also enforces `role === 'admin'` and the backend endpoints are gated, but a
// non-admin should never even reach the dashboard component, so this layout
// redirects first.
//
// Decision matrix (the /me response is the source of truth):
//   - role === 'admin'        -> render the dashboard.
//   - signed-out / 401        -> redirect to /login?next=... before the
//     dashboard loads.
//   - signed-in non-admin     -> redirect to the search home, NEVER /login:
//     sending them to /login would bounce the already-authenticated user back
//     here the moment the login page reads /me and auto-redirects — an
//     infinite login<->dashboard loop.
//   - network/5xx/timeout blip -> fail open to the client dashboard, which
//     re-checks with its own /me and mirrors this decision.
//
// WHY ABSOLUTE: the /me fetch runs inside the Next server. In same-origin
// production there is no NEXT_PUBLIC_API_BASE (nginx fronts the browser), so
// the browser's base is ''; but a RELATIVE '/api/auth/me' throws during Node's
// fetch URL parsing (Failed to parse URL) BEFORE any HTTP request is made — the
// guard would silently fail open on every render, and a next.config rewrite
// cannot help because the failure happens at URL-parse time, before a request
// exists to rewrite. The guard therefore builds its own ABSOLUTE base: the
// configured cross-origin base when one is set (dev, e2e scratch), else the
// backend loopback ($API_PORT, default 8001) that setup.sh and
// ecosystem.config.js both default to. This is server-only; the value never
// reaches the browser bundle. nginx still fronts the browser — the public data
// path is unchanged.

function serverMeBase(): string {
  const configured = readApiBaseEnv()
  if (configured) {
    const url = parseApiBaseUrl(configured)
    if (url) return url.origin + url.pathname.replace(/\/+$/, '')
  }
  const port = process.env.API_PORT || '8001'
  return `http://127.0.0.1:${port}`
}

const ME_TIMEOUT_MS = 8000

export default async function AnalyticsLayout({ children }: { children: ReactNode }) {
  // Three verdicts: definitive-admin, definitive-non-admin, definitive-signed-out
  // (all via status) and unverified (network/5xx). The 401 -> non-admin split
  // REQUIRES the status, so both the role and the verdict are kept.
  let role: string | null = null
  let signedOut = false
  let verified = false

  try {
    const cookieStore = await cookies()
    const res = await fetch(`${serverMeBase()}/api/auth/me`, {
      headers: { cookie: cookieStore.toString() },
      cache: 'no-store',
      signal: AbortSignal.timeout(ME_TIMEOUT_MS),
    })
    if (res.status === 200) {
      verified = true
      const body = (await res.json()) as { role?: unknown }
      role = typeof body.role === 'string' ? body.role : null
    } else if (res.status === 401) {
      verified = true
      signedOut = true
    }
    // Any other status (404/5xx from a proxy or the backend) stays unverified:
    // not a definitive auth verdict, so fail open rather than bounce an admin
    // to /login on a transient blip.
  } catch (err) {
    // Network/timeout blip: fail open to the client dashboard, which re-checks
    // with its own /me and mirrors the decision. An admin must not be bounced
    // to /login during an outage.
    console.error('[analytics] server guard could not reach /api/auth/me', err)
  }

  if (role === 'admin') {
    return <>{children}</>
  }

  if (!verified) {
    // Fail open to the client dashboard: its own /me is the authority and it
    // redirects/forbids before any analytics data is shown.
    return <>{children}</>
  }

  if (signedOut) {
    // `redirect()` throws NEXT_REDIRECT, which Next turns into a 307 Location
    // header; layouts return ReactNode, so NextResponse.redirect (only valid in
    // middleware/route handlers) is unavailable here.
    redirect(`/login?next=${encodeURIComponent('/analytics/dashboard')}`)
  }

  // verified + role present + not admin: a real signed-in non-admin. Away from
  // the dashboard, but NOT /login (that path loops — see the module docstring).
  redirect('/')
}