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
 *
 * All assertions anchor to global classes / roles / text, never CSS-module
 * hashed names (page.module.css classes are hashed at build time).
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
  await expect(page.getByRole('heading', { name: 'Feed', exact: true })).toBeVisible()

  // Add a subscription through the UI.
  await page.getByLabel('Add tag').fill('funding')
  await page.getByRole('button', { name: 'Add Tag' }).click()

  // The subscription shows as a removable chip (proof the POST round-tripped and
  // the list refetched). The chip is `chip <module-hashed>` — anchor on the
  // global `.chip` class plus its text.
  const chip = page.locator('.chip', { hasText: 'funding' })
  await expect(chip).toBeVisible({ timeout: 20_000 })

  // The feed refetched after the subscription: it renders either a matching
  // article link (a[href^="http"], the same anchor for-you.spec uses) or the
  // backend's honest empty note — never an error state. Poll on visible text,
  // not hashed classes.
  await expect
    .poll(
      async () => {
        const hasCard = await page.locator('a[href^="http"]').first().isVisible().catch(() => false)
        const hasEmpty = await page
          .getByText(/No articles match your subscriptions yet|not following anything yet/)
          .isVisible()
          .catch(() => false)
        const hasError = await page.getByRole('alert').isVisible().catch(() => false)
        return hasError ? 'error' : hasCard || hasEmpty ? 'ok' : 'loading'
      },
      { timeout: 20_000 },
    )
    .toBe('ok')
})
