import { test, expect, type Browser } from '@playwright/test'
import { login, signOut, signUpFreshUser, uniqueEmail } from './support'


test.describe('auth journey', () => {
  const email = uniqueEmail('e2e-auth')

  test('signup, logout, login, protected route', async ({ page }) => {
    
    await signUpFreshUser(page, email)

    
    await signOut(page)
    await expect(page).toHaveURL(/\/login/, { timeout: 20_000 })
    await expect(page.locator('button.topbar-logout')).toHaveCount(0)

    
    
    await login(page, email)

    
    
    await page.goto('/for-you')
    await expect(page.getByRole('heading', { name: 'For You' })).toBeVisible()
    
    await expect(page.locator('a[href^="http"]').first()).toBeVisible({ timeout: 30_000 })
  })

  test('protected /for-you answers 401 without a session cookie', async ({
    browser,
  }) => {
    await assertForYouUnauthenticated(browser, email)
  })
})

async function assertForYouUnauthenticated(browser: Browser, email: string): Promise<void> {
  
  const ctx = await browser.newContext()
  const page = await ctx.newPage()
  try {
    const response = await page.goto('/for-you')
    
    expect(response?.status()).toBe(200)
    await expect(page.getByRole('heading', { name: 'For You' })).toBeVisible()
    
    
    
    const errPanel = page.locator('div').filter({ hasText: 'Failed to load feed: 401' }).last()
    await expect(errPanel).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
  } finally {
    await ctx.close()
  }
}
