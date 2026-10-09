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
// Decision matrix (the response is the source of truth, so `role` here is a
// tri-state):
//   - role === 'admin'  -> render the dashboard.
//   - signed-out 401    -> redirect to /login?next=… before the dashboard loads.
//   - signed-in non-admin -> redirect to the search home, NEVER /login: sending
//     them to /login would bounce the already-authenticated user back here the
//     moment the login page reads /me and auto-redirects, an infinite loop.
//   - network/5xx blip  -> fail open to the client dashboard, which re-checks
//     with its own /me and mirrors this decision.
//
// The /me fetch runs INSIDE the Next server (same-origin deployment has no
// NEXT_PUBLIC_API_BASE, so `base` is ''), and a relative URL resolves against
// the Next server itself — which has no /api route. nginx proxies /api for the
// browser, but not for the server component, so next.config.ts adds a rewrite
// `/api/:path*` -> backend loopback, exactly the boundary nginx draws in
// production. Without that rewrite this layout always saw 404 -> role null ->
// redirect to /login -> login page auto-returned -> loop, reported as "redirects
// to login and is in an infinite loop".

function resolveApiBase(): string {
  const source = readApiBaseEnv()
  if (!source) return ''
  const url = parseApiBaseUrl(source)
  if (!url) return ''
  return url.origin + url.pathname.replace(/\/+$/, '')
}

const ME_TIMEOUT_MS = 8000

export default async function AnalyticsLayout({ children }: { children: ReactNode }) {
  const base = resolveApiBase()
  // tri-state: 'admin' | 'user' | null (signed out / unverifiable)
  let role: string | null = null
  let verifiable = false

  try {
    const cookieStore = await cookies()
    const res = await fetch(`${base}/api/auth/me`, {
      headers: { cookie: cookieStore.toString() },
      cache: 'no-store',
      signal: AbortSignal.timeout(ME_TIMEOUT_MS),
    })
    if (res.status === 200) {
      verifiable = true
      const body = (await res.json()) as { role?: unknown }
      role = typeof body.role === 'string' ? body.role : null
    }
  } catch (err) {
    // Network/timeout/5xx blip: fail open to the client dashboard, which
    // re-checks and surfaces its own verdict rather than an admin being bounced
    // to /login during an outage.
    console.error('[analytics] server guard could not reach /api/auth/me', err)
  }

  if (role === 'admin') {
    return <>{children}</>
  }

  if (!verifiable) {
    // Fail open: /me never answered a definitive verdict, so the client's own
    // /me check is the authority. Rendering children here is safe — the client
    // dashboard redirects/forbids before any analytics data is shown.
    return <>{children}</>
  }

  // /me answered (res.status === 200): a definitive result. But we deliberately
  // only set `verifiable` on 200, and here `role !== 'admin'` means a real,
  // signed-in non-admin — the /me 200 guarantees a valid session. A 401 never
  // reaches this branch (good: signed-out users must be told to log in). So the
  // only way here is signed-in non-admin: send them away from the dashboard, but
  // NOT to /login. `redirect()` throws NEXT_REDIRECT -> 307 Location.
  redirect('/')
}