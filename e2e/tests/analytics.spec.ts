import { test, expect } from '@playwright/test'
import { API_BASE, login } from './support'

const ADMIN_EMAIL = process.env.E2E_ADMIN_EMAIL
const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD


test('unauthenticated /analytics/dashboard redirects to /login', async ({ page }) => {
  await page.goto('/analytics/dashboard')
  await expect(page).toHaveURL(/\/login/, { timeout: 20_000 })
})

test('admin sees real analytics fed by backend endpoints', async ({ page }) => {
  if (!ADMIN_EMAIL || !ADMIN_PASSWORD) {
    test.skip(true, 'E2E_ADMIN_EMAIL/PASSWORD not provided — cannot log in as admin')
  }
  await login(page, ADMIN_EMAIL!, ADMIN_PASSWORD!)

  
  
  const q = process.env.E2E_SEARCH_QUERY || 'Ola Electric IPO'
  const search = await page.request.get(`${API_BASE}/search?q=${encodeURIComponent(q)}`)
  expect(search.status()).toBe(200)
  const clicks = await page.request.post(`${API_BASE}/analytics/click`, {
    data: { query: q, position: 1, id: 1 },
  })
  expect([200, 202]).toContain(clicks.status())

  await page.goto('/analytics/dashboard')
  
  
  
  
  await expect(page.getByRole('heading', { name: 'Top queries' })).toBeVisible()
  await expect(page.getByRole('heading', { name: /Chat usage/ })).toBeVisible()
  await expect(page.locator('.dash-cards .dash-card').first()).toBeVisible()
})
