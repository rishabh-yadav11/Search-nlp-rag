/**
 * The top bar is the app's only chrome and every route mounts it, so its
 * shared behaviour lives here: the active nav item follows the route, the
 * Analytics link is admin-gated, the account slot is tri-state (unknown /
 * signed in / signed out) and the session is only fetched when the caller does
 * not already know it.
 */
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TopBar from './TopBar'
import type { AuthUser } from '../lib/auth'

const h = vi.hoisted(() => ({
  pathname: { current: '/' },
  getMe: vi.fn<() => Promise<AuthUser | null>>(),
  logout: vi.fn(),
}))

vi.mock('next/navigation', () => ({ usePathname: () => h.pathname.current }))
vi.mock('../lib/auth', () => ({ getMe: h.getMe, logout: h.logout }))

const USER: AuthUser = {
  id: 'u1',
  email: 'rep@vccircle.com',
  name: 'Rep',
  role: 'user',
  is_active: true,
}
const ADMIN: AuthUser = { ...USER, id: 'a1', email: 'boss@vccircle.com', name: 'Boss', role: 'admin' }

const ROUTES: { path: string; label: string }[] = [
  { path: '/', label: 'Search' },
  { path: '/for-you', label: 'For You' },
  { path: '/chat', label: 'Chat' },
]

function activeLinks(): string[] {
  return [...document.querySelectorAll('.topbar-nav-link.active')].map((el) => el.textContent ?? '')
}

beforeEach(() => {
  h.pathname.current = '/'
  h.getMe.mockReset()
  h.logout.mockReset()
  h.getMe.mockResolvedValue(null)
})

describe('TopBar nav', () => {
  it.each(ROUTES)('marks $label active on $path', ({ path, label }) => {
    h.pathname.current = path
    render(<TopBar me={null} />)
    expect(activeLinks()).toEqual([label])
  })

  it('marks Analytics active on the dashboard for an admin', () => {
    h.pathname.current = '/analytics/dashboard'
    render(<TopBar me={ADMIN} />)
    expect(activeLinks()).toEqual(['Analytics'])
  })

  it('marks nothing active on a route that is not in the nav', () => {
    h.pathname.current = '/login'
    render(<TopBar me={ADMIN} />)
    expect(activeLinks()).toEqual([])
  })

  it('offers Analytics to an admin only', () => {
    const { unmount } = render(<TopBar me={USER} />)
    expect(screen.queryByRole('link', { name: 'Analytics' })).toBeNull()
    unmount()

    render(<TopBar me={ADMIN} />)
    expect(screen.getByRole('link', { name: 'Analytics' }).getAttribute('href')).toBe(
      '/analytics/dashboard'
    )
  })

  it('keeps the ASK VCCircle call to action on every other route', () => {
    for (const { path } of ROUTES.filter((r) => r.path !== '/chat')) {
      const view = render(<TopBar me={null} />)
      h.pathname.current = path
      view.rerender(<TopBar me={null} />)
      expect(screen.getByRole('link', { name: 'Open chat assistant' }).getAttribute('href')).toBe('/chat')
      view.unmount()
    }
  })

  it('drops the call to action on /chat, where it would link to itself', () => {
    h.pathname.current = '/chat'
    render(<TopBar me={null} />)
    expect(screen.queryByRole('link', { name: 'Open chat assistant' })).toBeNull()
    // The nav item still marks the route, so nothing is lost.
    expect(screen.getByRole('link', { name: 'Chat' }).className).toContain('active')
  })
})

describe('TopBar account control', () => {
  it('offers Sign in when the session is known to be absent', () => {
    render(<TopBar me={null} />)
    expect(screen.getByRole('link', { name: 'Sign in' }).getAttribute('href')).toBe('/login')
    expect(screen.queryByRole('button', { name: 'Log out' })).toBeNull()
  })

  it('shows the signed-in user and a Log out button', () => {
    render(<TopBar me={USER} />)
    expect(screen.getByText(USER.name)).toBeTruthy()
    expect(screen.getByTitle(USER.email)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Log out' })).toBeTruthy()
    expect(screen.queryByRole('link', { name: 'Sign in' })).toBeNull()
  })

  it('labels the account with the email when the user has no name', () => {
    render(<TopBar me={{ ...USER, name: '' }} />)
    expect(screen.getByText(USER.email)).toBeTruthy()
  })

  it('shows nothing at all while the session is still unknown', () => {
    // Never settles: this is the loading state the account slot must tolerate.
    h.getMe.mockReturnValue(new Promise<AuthUser | null>(() => {}))
    render(<TopBar />)
    expect(screen.queryByRole('link', { name: 'Sign in' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Log out' })).toBeNull()
  })
})

describe('TopBar session lookup', () => {
  it('never asks for the session when the caller already knows it', () => {
    render(<TopBar me={USER} />)
    expect(h.getMe).not.toHaveBeenCalled()
  })

  it('never asks for the session while the caller is still resolving it', () => {
    // The bar treats the PROP's presence, not its value, as "the caller owns
    // this request". A page that seeds its own identity state with `undefined`
    // is mid-fetch, not asking the bar to fetch a second time.
    render(<TopBar me={undefined} />)
    expect(h.getMe).not.toHaveBeenCalled()
    expect(screen.queryByRole('link', { name: 'Sign in' })).toBeNull()
  })

  it('adopts the caller session as soon as it lands', () => {
    const { rerender } = render(<TopBar me={undefined} />)
    expect(screen.queryByTitle(USER.email)).toBeNull()
    rerender(<TopBar me={USER} />)
    expect(screen.getByTitle(USER.email)).toBeTruthy()
  })

  it('asks for the session once when no value is passed', async () => {
    h.getMe.mockResolvedValue(USER)
    render(<TopBar />)
    expect(await screen.findByRole('button', { name: 'Log out' })).toBeTruthy()
    expect(h.getMe).toHaveBeenCalledTimes(1)
  })

  it('shows the admin nav once the fetched session resolves', async () => {
    h.getMe.mockResolvedValue(ADMIN)
    h.pathname.current = '/analytics/dashboard'
    render(<TopBar />)
    expect(await screen.findByRole('link', { name: 'Analytics' })).toBeTruthy()
    expect(activeLinks()).toEqual(['Analytics'])
  })

  it('falls back to signed out when the session check rejects', async () => {
    h.getMe.mockRejectedValue(new Error('offline'))
    render(<TopBar />)
    expect(await screen.findByRole('link', { name: 'Sign in' })).toBeTruthy()
  })
})

describe('TopBar subtitle', () => {
  it('renders the subtitle inside the brand next to the wordmark', () => {
    render(<TopBar me={null} subtitle="Conversation" />)
    const subtitle = screen.getByText('Conversation')
    expect(subtitle.className).toBe('topbar-subtitle')
    expect(subtitle.closest('.brand')).not.toBeNull()
  })

  it('omits the subtitle when none is passed', () => {
    const { container } = render(<TopBar me={null} />)
    expect(container.querySelector('.topbar-subtitle')).toBeNull()
  })

  it('omits the subtitle when it is empty', () => {
    const { container } = render(<TopBar me={null} subtitle="" />)
    expect(container.querySelector('.topbar-subtitle')).toBeNull()
  })
})
