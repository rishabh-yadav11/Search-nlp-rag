import { test, expect } from '@playwright/test'
import { signUpFreshUser, uniqueEmail } from './support'

/**
 * /feed is auth-gated (backend require_auth on /api/feed/*). An anonymous
 * visitor's first feed load 401s and the page redirects to /login preserving
 * next. After sign-in, adding a subscription through the UI round-trips to the
 * backend and the feed refetches: it renders either matching cards or the
 * backend's honest empty note — never an error state. The scratch corpus is the
 * real trimmed production corpus, so exact facet values are unknown here; the
 * subscription value below is corpus-independent and only proves the
 * add/list/feed-refetch pipeline, not content correctness.
 */
test('unauthenticated /feed redirects to login preserving next', async ({ page }) => {
  await page.goto('/feed')
  await expect(page).toHaveURL(/\/login/, { timeout: 20_000 })
  // The login page carries the intended destination back.
  await expect(page).toHaveURL(/next=%2Ffeed/)
})

test('signed-in user can subscribe and gets a feed view', async ({ page }) => {
  await signUpFreshUser(page, uniqueEmail('e2e-feed'))

  await page.goto('/feed')
  await expect(page.getByRole('heading', { name: 'Feed' })).toBeVisible()

  // Add a subscription through the UI (id persists as the datalist id).
  await page.getByLabel('Add tag').fill('funding')
  await page.getByRole('button', { name: 'Add Tag' }).click()

  // The subscription shows as a removable chip (proof the POST round-tripped and
  // the list refetched from the single source of truth).
  const chip = page.locator('.subscription-chip', { hasText: 'funding' })
  await expect(chip).toBeVisible({ timeout: 20_000 })

  // The feed refetched after the subscription: it renders either matching
  // article cards or the backend's honest empty note — never an error.
  await expect
    .poll(
      async () => {
        const errorVisible = await page.locator('.feed-page .error').isVisible().catch(() => false)
        const emptyVisible = await page.locator('.feed-page .empty').isVisible().catch(() => false)
        const cardVisible = await page.locator('.article-card').first().isVisible().catch(() => false)
        return errorVisible ? 'error' : emptyVisible ? 'empty' : cardVisible ? 'cards' : 'neither'
      },
      { timeout: 20_000 },
    )
    .not.toBe('error')
  await expect(page.getByText('Your feed', { exact: true })).toBeVisible()
})
