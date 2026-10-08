import { test, expect, type Browser } from '@playwright/test'
import { login, signOut, signUpFreshUser, uniqueEmail } from './support'

/**
 * Auth journey end-to-end against the real /api/auth/* endpoints:
 *
 *   signup -> land on /chat (default next) -> logout -> login -> protected /for-you
 *
 * plus the auth gate: /for-you is backend-401 without a session cookie (the
 * backend enforces require_auth on /recommend/for-you; the frontend surfaces the
 * 401 as "Failed to load feed: 401" with a Retry button, it never redirects away).
 */
test.describe('auth journey', () => {
  const email = uniqueEmail('e2e-auth')

  test('signup, logout, login, protected route', async ({ page }) => {
    // 1) Sign up a brand-new user -> lands on /chat, TopBar shows account control.
    await signUpFreshUser(page, email)

    // 2) Logout -> redirected to /login and the account control disappears.
    await signOut(page)
    await expect(page).toHaveURL(/\/login/, { timeout: 20_000 })
    await expect(page.locator('button.topbar-logout')).toHaveCount(0)

    // 3) Login with the same credentials -> /chat again (proves the saved hash
    //    verifies), account control back.
    await login(page, email)

    // 4) Protected route WITH the session cookie: /for-you fetches real
    //    recommendations.
    await page.goto('/for-you')
    await expect(page.getByRole('heading', { name: 'For You' })).toBeVisible()
    // The feed loads real recommendation cards = external article links.
    await expect(page.locator('a[href^="http"]').first()).toBeVisible({ timeout: 30_000 })
  })

  test('protected /for-you answers 401 without a session cookie', async ({
    browser,
  }) => {
    await assertForYouUnauthenticated(browser, email)
  })
})

async function assertForYouUnauthenticated(browser: Browser, email: string): Promise<void> {
  // A clean context (no cookies) is the unauthenticated client.
  const ctx = await browser.newContext()
  const page = await ctx.newPage()
  try {
    const response = await page.goto('/for-you')
    // Page itself is public static shell.
    expect(response?.status()).toBe(200)
    await expect(page.getByRole('heading', { name: 'For You' })).toBeVisible()
    // The feed fetch 401s; the page shows the honest error rather than data.
    // The error text sits in both the page wrapper and the error panel, so
    // query the error panel itself (the last matching div).
    const errPanel = page.locator('div').filter({ hasText: 'Failed to load feed: 401' }).last()
    await expect(errPanel).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
  } finally {
    await ctx.close()
  }
}
