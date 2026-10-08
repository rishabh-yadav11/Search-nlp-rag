import { test, expect } from '@playwright/test'
import { API_BASE, login } from './support'

const ADMIN_EMAIL = process.env.E2E_ADMIN_EMAIL
const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD

/**
 * /analytics/dashboard is admin-only (backend require_permission("analytics:read")).
 * The scratch backend seeds a bootstrap admin from AUTH_ADMIN_* (run-on-box.sh sets
 * E2E_ADMIN_*). Unauthenticated, the dashboard client-side redirects to /login.
 */
test('unauthenticated /analytics/dashboard redirects to /login', async ({ page }) => {
  await page.goto('/analytics/dashboard')
  await expect(page).toHaveURL(/\/login/, { timeout: 20_000 })
})

test('admin sees real analytics fed by backend endpoints', async ({ page }) => {
  if (!ADMIN_EMAIL || !ADMIN_PASSWORD) {
    test.skip(true, 'E2E_ADMIN_EMAIL/PASSWORD not provided — cannot log in as admin')
  }
  await login(page, ADMIN_EMAIL!, ADMIN_PASSWORD!)

  // Seed the analytics store through the REAL public endpoints so the dashboard
  // has live figures rather than an all-zeros first paint.
  const q = process.env.E2E_SEARCH_QUERY || 'Ola Electric IPO'
  const search = await page.request.get(`${API_BASE}/search?q=${encodeURIComponent(q)}`)
  expect(search.status()).toBe(200)
  const clicks = await page.request.post(`${API_BASE}/analytics/click`, {
    data: { query: q, position: 1, id: 1 },
  })
  expect([200, 202]).toContain(clicks.status())

  await page.goto('/analytics/dashboard')
  // The dashboard's success state has no "Search analytics" heading: it shows a
  // stat-card grid plus "Top queries" and "Chat usage" panels, so assert on
  // what a permitted admin actually sees (proves /analytics/summary + /chat
  // loaded through the real API).
  await expect(page.getByRole('heading', { name: 'Top queries' })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Chat usage/ })).toBeVisible()
  await expect(page.locator('.dash-cards .dash-card').first()).toBeVisible()
})
