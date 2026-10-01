'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { AuthUser, getMe, logout } from '../lib/auth'

// The active item is derived from the route so every page that mounts this bar
// highlights itself.
const NAV: { href: string; label: string }[] = [
  { href: '/', label: 'Search' },
  { href: '/for-you', label: 'For You' },
  { href: '/chat', label: 'Chat' },
]

// Admin-only route, excluded from the always-on nav list.
const ANALYTICS_HREF = '/analytics/dashboard'

/**
 * `me` is tri-state: `undefined` renders nothing in the account slot, `null` is
 * signed out. Whether the bar resolves the session ITSELF is decided by the
 * PRESENCE of the prop, not its value, so a page that already owns the request
 * can hand over a still-loading `undefined` without this bar firing a second,
 * concurrent `/api/auth/me` (`getMe` caches results, not in-flight calls).
 */
export default function TopBar(props: { me?: AuthUser | null; subtitle?: string | null }) {
  const { me: meProp, subtitle } = props
  const callerOwnsSession = 'me' in props
  const pathname = usePathname()
  const [fetchedMe, setFetchedMe] = useState<AuthUser | null | undefined>(undefined)

  useEffect(() => {
    // A caller-supplied value wins, loading or not: never re-ask for a session
    // the page is already fetching.
    if (callerOwnsSession) return
    let live = true
    getMe()
      .then((u) => {
        if (live) setFetchedMe(u)
      })
      .catch(() => {
        // A failed session check means signed out, not "still loading".
        if (live) setFetchedMe(null)
      })
    return () => {
      live = false
    }
  }, [callerOwnsSession])

  const me = callerOwnsSession ? meProp : fetchedMe

  const links = me?.role === 'admin' ? [...NAV, { href: ANALYTICS_HREF, label: 'Analytics' }] : NAV

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="logo" src="/vccircle-wordmark.svg" alt="VCCircle" width={154} height={40} />
          {subtitle ? <span className="topbar-subtitle">{subtitle}</span> : null}
        </div>
        <nav className="topbar-nav" aria-label="Primary">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={`topbar-nav-link${pathname === l.href ? ' active' : ''}`}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="topbar-right">
          {/* On /chat this CTA would link to the page already on screen, and the
              nav's "Chat" item already marks the route as current. */}
          {pathname !== '/chat' ? (
            <Link href="/chat" className="topbar-cta" aria-label="Open chat assistant">
              ASK VCCircle
            </Link>
          ) : null}
          {me === undefined ? null : me ? (
            <span className="topbar-user">
              <span className="topbar-user-email" title={me.email}>
                {me.name || me.email}
              </span>
              <button type="button" className="topbar-logout" onClick={logout}>
                Log out
              </button>
            </span>
          ) : (
            <Link href="/login" className="topbar-signin">
              Sign in
            </Link>
          )}
        </div>
      </div>
    </header>
  )
}
