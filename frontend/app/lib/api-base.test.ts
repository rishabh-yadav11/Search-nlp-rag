import { NextRequest } from 'next/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { middleware } from '../../middleware'
import { DEV_LOOPBACK, devApiBase, parseApiBaseUrl, readApiBaseEnv, sanitizeApiBaseOrigin } from './api-base'

/**
 * This file also covers `frontend/middleware.ts`. The test lives under `app/`
 * only because `vitest.config.ts` restricts its include glob to `app/`; nothing
 * about the middleware depends on being in the app directory.
 *
 * The point of the middleware cases is that the edge bundle and the browser
 * bundle now share one validator and one `NEXT_PUBLIC_API_BASE` read. They used
 * to carry separate copies that disagreed, so these assert the value the
 * operator configured actually reaches the CSP — end to end, through the real
 * `middleware()` call rather than through a copy of its logic.
 */

const ORIGIN = 'https://app.example.com'

afterEach(() => {
  vi.unstubAllEnvs()
})

function cspFor(env: Record<string, string | undefined>): string {
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) vi.stubEnv(key, '')
    else vi.stubEnv(key, value)
  }
  const res = middleware(new NextRequest('http://localhost:3000/chat'))
  return res.headers.get('content-security-policy') ?? ''
}

function connectSrcOf(csp: string): string[] {
  const line = csp.split('; ').find((d) => d.startsWith('connect-src'))
  return (line ?? '').split(' ').slice(1)
}

describe('parseApiBaseUrl — accepts ordinary bases', () => {
  const OK = [
    'https://api.example.com',
    'http://api.example.com',
    'https://api.example.com/v1',
    'https://api.example.com:8443',
    'http://localhost:8001',
    'https://api.example.com/',
  ]

  it.each(OK)('parses %j', (value) => {
    expect(parseApiBaseUrl(value)).toBeInstanceOf(URL)
  })

  it('keeps the path, which a request base needs', () => {
    expect(parseApiBaseUrl('https://api.example.com/v1')?.pathname).toBe('/v1')
  })

  it('resolves a root-relative base only when given an origin to resolve against', () => {
    expect(parseApiBaseUrl('/api', ORIGIN)?.href).toBe('https://app.example.com/api')
    // The edge runtime has no current origin, so a relative base is unusable
    // there and must be refused rather than resolved against something.
    expect(parseApiBaseUrl('/api')).toBeNull()
  })
})

describe('parseApiBaseUrl — refuses bases that are not plain http(s) origins', () => {
  const REJECTED = [
    'javascript:alert(1)',
    'data:text/html,x',
    'file:///etc/passwd',
    '//evil.com',
    'https://user:pw@api.example.com',
    'https://user@api.example.com',
    'https://api.example.com; script-src *',
    'https://api example.com',
    'https://api.example.com/a b',
    'https://api.example.com\n',
    'not a url',
    '',
    null,
    undefined,
    42,
  ]

  it.each(REJECTED)('rejects %j', (value) => {
    expect(parseApiBaseUrl(value)).toBeNull()
  })
})

describe('sanitizeApiBaseOrigin — projects a bare origin for connect-src', () => {
  it('drops the path, because connect-src is an origin list', () => {
    expect(sanitizeApiBaseOrigin('https://api.example.com/v1')).toBe('https://api.example.com')
  })
})

describe('devApiBase — loopback only in development', () => {
  it('yields the loopback origin in development', () => {
    vi.stubEnv('NODE_ENV', 'development')
    expect(devApiBase()).toBe(DEV_LOOPBACK)
    expect(DEV_LOOPBACK).toBe('http://localhost:8001')
  })

  it.each(['production', 'test'])('yields nothing in %s', (env) => {
    vi.stubEnv('NODE_ENV', env)
    expect(devApiBase()).toBe('')
  })
})

describe('readApiBaseEnv — the single configured-base read', () => {
  it('yields an empty base when unset', () => {
    vi.stubEnv('NEXT_PUBLIC_API_BASE', '')
    expect(readApiBaseEnv()).toBe('')
  })

  it('yields the configured value when set', () => {
    vi.stubEnv('NEXT_PUBLIC_API_BASE', 'https://api.example.com')
    expect(readApiBaseEnv()).toBe('https://api.example.com')
  })
})

describe('middleware — the configured base reaches connect-src', () => {
  it('whitelists the configured origin', () => {
    const csp = cspFor({ NEXT_PUBLIC_API_BASE: 'https://api.example.com/v1', NODE_ENV: 'production' })
    expect(connectSrcOf(csp)).toEqual(["'self'", 'https://api.example.com'])
  })

  it('never emits a path or a non-origin into the directive', () => {
    const csp = cspFor({ NEXT_PUBLIC_API_BASE: 'https://api.example.com/v1', NODE_ENV: 'production' })
    expect(csp).not.toContain('/v1')
  })

  it.each([
    ['an injected directive', "https://api.example.com'; script-src 'unsafe-inline'"],
    ['a script scheme', 'javascript:alert(1)'],
    ['embedded credentials', 'https://user:pw@api.example.com'],
  ])('emits no extra connect-src source for %s', (_label, value) => {
    const csp = cspFor({ NEXT_PUBLIC_API_BASE: value, NODE_ENV: 'production' })
    expect(connectSrcOf(csp)).toEqual(["'self'"])
  })

  it('locks production to self when no base is configured', () => {
    const csp = cspFor({ NEXT_PUBLIC_API_BASE: '', NODE_ENV: 'production' })
    expect(connectSrcOf(csp)).toEqual(["'self'"])
  })

  it('falls back to the dev loopback only in development', () => {
    expect(connectSrcOf(cspFor({ NEXT_PUBLIC_API_BASE: '', NODE_ENV: 'development' }))).toEqual([
      "'self'",
      DEV_LOOPBACK,
    ])
  })

  it('prefers the configured base over the dev loopback', () => {
    const csp = cspFor({ NEXT_PUBLIC_API_BASE: 'https://api.example.com', NODE_ENV: 'development' })
    expect(connectSrcOf(csp)).toEqual(["'self'", 'https://api.example.com'])
  })
})
