'use client'
import { devApiBase, parseApiBaseUrl, readApiBaseEnv } from './api-base'
import { isSafeRedirect } from './safe-url'
import { createDeadline, LOGOUT_DEADLINE_MS, ME_DEADLINE_MS } from './deadline'




const TRUSTED_API_HOSTS: string[] = (process.env.NEXT_PUBLIC_TRUSTED_API_HOSTS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)










const ENV_API_BASE = readApiBaseEnv()
const DEV_API_BASE = devApiBase()



function isSameOrigin(url: URL): boolean {
  if (typeof window === 'undefined') return false
  return url.origin === window.location.origin
}


function isHttps(url: URL): boolean {
  return url.protocol === 'https:'
}


function isLoopback(url: URL): boolean {
  
  return ['localhost', '127.0.0.1', '::1'].includes(url.hostname)
}


export function normalizeHost(host: string): string {
  return host
    .replace(/^https?:\/\//, '')
    .replace(/\/+$/, '')
    .replace(/\.$/, '')
    .replace(/:(?:80|443)$/, '')
    .toLowerCase()
}


function hostInAllowList(url: URL): boolean {
  const h = normalizeHost(url.host)
  return TRUSTED_API_HOSTS.some((entry) => normalizeHost(entry) === h)
}


function baseFromUrl(url: URL): string {
  
  
  
  return url.origin + url.pathname.replace(/\/+$/, '')
}


function isTrustedBase(base: string, url: URL | null): boolean {
  if (!base) return true 
  if (!url) return false
  if (isSameOrigin(url)) return true 
  
  if (isLoopback(url)) return true
  
  return isHttps(url) && hostInAllowList(url)
}

function resolveApiBase(): { base: string; trusted: boolean } {
  
  
  const currentOrigin = typeof window !== 'undefined' ? window.location.origin : undefined
  
  
  
  const winApiBase =
    (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) || ''
  if (winApiBase) {
    const url = parseApiBaseUrl(winApiBase, currentOrigin)
    if (!url) {
      console.error(
        '[auth] window.API_BASE is not a valid http(s) URL; ignoring it and using the safe same-origin base.'
      )
      return { base: '', trusted: true }
    }
    if (isTrustedBase(winApiBase, url)) {
      return { base: baseFromUrl(url), trusted: true }
    }
    console.error(
      `[auth] window.API_BASE host "${normalizeHost(url.host)}" is not same-origin and not in NEXT_PUBLIC_TRUSTED_API_HOSTS; using the safe same-origin base instead.`
    )
    return { base: '', trusted: true }
  }
  
  
  const trustedSource = ENV_API_BASE || DEV_API_BASE
  if (trustedSource) {
    const url = parseApiBaseUrl(trustedSource, currentOrigin)
    if (!url) {
      console.error(
        `[auth] Configured API base "${trustedSource}" is not a valid http(s) URL; falling back to the safe same-origin base.`
      )
      return { base: '', trusted: true }
    }
    const trusted = isTrustedBase(trustedSource, url)
    if (!trusted) {
      console.error(
        `[auth] Configured API base "${trustedSource}" is cross-origin and not trusted (https + NEXT_PUBLIC_TRUSTED_API_HOSTS required); the session cookie will NOT be sent.`
      )
    }
    return { base: baseFromUrl(url), trusted }
  }
  
  return { base: '', trusted: true }
}

const RESOLVED_API_BASE = resolveApiBase()


export const API_BASE = RESOLVED_API_BASE.base

export const API_BASE_TRUSTED = RESOLVED_API_BASE.trusted



const LEGACY_TOKEN_KEY = 'vccircle_auth_token'


export function clearLegacyToken(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(LEGACY_TOKEN_KEY)
  } catch {}
}


export function authRequestInit(init?: RequestInit): RequestInit {
  if (!API_BASE_TRUSTED) return init ?? {}
  return { ...init, credentials: 'include' }
}

export interface AuthUser {
  id: string
  email: string
  name: string
  role: string
  is_active: boolean
}

let meCache: AuthUser | null | undefined
let meCacheTs = 0


const ME_CACHE_TTL_RAW = Number(process.env.NEXT_PUBLIC_ME_CACHE_TTL_MS || 60000)
const ME_CACHE_TTL_MS = Number.isNaN(ME_CACHE_TTL_RAW) ? 60000 : ME_CACHE_TTL_RAW


export async function getMe(force = false, signal?: AbortSignal | null): Promise<AuthUser | null> {
  
  clearLegacyToken()
  const fresh = meCache !== undefined && (ME_CACHE_TTL_MS <= 0 || Date.now() - meCacheTs < ME_CACHE_TTL_MS)
  if (!force && fresh) return meCache ?? null
  
  
  
  
  
  const deadline = createDeadline(ME_DEADLINE_MS, signal ?? null)
  try {
    
    
    const res = await fetch(
      `${API_BASE}/api/auth/me`,
      authRequestInit({ signal: deadline.signal })
    )
    if (res.status === 401) {
      
      clearMeCache()
      clearLegacyToken()
      return null
    }
    if (!res.ok) return null
    try {
      meCache = (await res.json()) as AuthUser
    } catch (err) {
      
      
      
      
      if (deadline.timedOut() || (err as Error)?.name === 'AbortError') throw err
      
      console.error('getMe: failed to parse /api/auth/me response')
      return null
    }
  } catch (err) {
    
    
    
    
    
    if (!deadline.timedOut() && (err as Error)?.name !== 'AbortError') {
      console.error('getMe: failed to reach the auth service', err)
    } else if (deadline.timedOut()) {
      console.error('getMe: /api/auth/me timed out', err)
    }
    throw err
  } finally {
    deadline.clear()
  }
  meCacheTs = Date.now()
  return meCache
}

export function clearMeCache(): void {
  meCache = undefined
  meCacheTs = 0
}


export function logout(): void {
  const deadline = createDeadline(LOGOUT_DEADLINE_MS)
  fetch(
    `${API_BASE}/api/auth/logout`,
    authRequestInit({ method: 'POST', keepalive: true, signal: deadline.signal })
  )
    .catch(() => {})
    .finally(() => deadline.clear())
  redirectToLogin()
}


export function redirectToLogin(next?: string): void {
  clearMeCache()
  if (typeof window !== 'undefined') {
    
    const target = isSafeRedirect(next) ? `/login?next=${encodeURIComponent(next)}` : '/login'
    window.location.replace(target)
  }
}