import { test, expect } from '@playwright/test'
import { signUpFreshUser, uniqueEmail } from './support'


test('unauthenticated /feed redirects to login preserving next', async ({ page }) => {
  await page.goto('/feed')
  await expect(page).toHaveURL(/\/login/, { timeout: 20_000 })
  
  await expect(page).toHaveURL(/next=%2Ffeed/)
})

test('signed-in user can subscribe and gets a feed view', async ({ page }) => {
  await signUpFreshUser(page, uniqueEmail('e2e-feed'))

  await page.goto('/feed')
  await expect(page.getByRole('heading', { name: 'Feed', exact: true })).toBeVisible()

  
  await page.getByLabel('Add tag').fill('funding')
  await page.getByRole('button', { name: 'Add Tag' }).click()

  
  
  
  const chip = page.locator('.chip', { hasText: 'funding' })
  await expect(chip).toBeVisible({ timeout: 20_000 })

  
  
  
  
  
  
  await expect
    .poll(
      async () => {
        const hasCard = await page.locator('a[href^="http"]').first().isVisible().catch(() => false)
        const hasEmpty = await page
          .getByText(/No articles match your subscriptions yet|not following anything yet/)
          .isVisible()
          .catch(() => false)
        const hasRetry = await page.getByRole('button', { name: 'Retry' }).isVisible().catch(() => false)
        return hasRetry ? 'error' : hasCard || hasEmpty ? 'ok' : 'loading'
      },
      { timeout: 20_000 },
    )
    .toBe('ok')
})
