import { test, expect } from '@playwright/test'
import { signUpFreshUser, uniqueEmail } from './support'


test('for-you renders recommendation cards after login', async ({ page }) => {
  await signUpFreshUser(page, uniqueEmail('e2e-foryou'))

  await page.goto('/for-you')
  await expect(page.getByRole('heading', { name: 'For You' })).toBeVisible()

  
  
  await page.getByRole('button', { name: 'Trending' }).click()

  
  
  const cardLink = page.locator('a[href^="http"]').first()
  await expect(cardLink).toBeVisible({ timeout: 30_000 })
  
  await expect(cardLink).toHaveText(/.+/)
})
