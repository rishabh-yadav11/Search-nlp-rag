import type { ReactNode } from 'react'
import { redirect } from 'next/navigation'
import { cookies } from 'next/headers'
import { readApiBaseEnv, parseApiBaseUrl } from '../lib/api-base'

export const dynamic = 'force-dynamic'






























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
  } catch (err) {
    
    
    
    console.error('[analytics] server guard could not reach /api/auth/me', err)
  }

  if (role === 'admin') {
    return <>{children}</>
  }

  if (!verified) {
    
    
    return <>{children}</>
  }

  if (signedOut) {
    
    
    
    redirect(`/login?next=${encodeURIComponent('/analytics/dashboard')}`)
  }

  
  
  redirect('/')
}