import { test, expect } from '@playwright/test'
import { signUpFreshUser, uniqueEmail } from './support'

/**
 * /for-you is auth-gated (backend require_auth on /recommend/*). After signing
 * in, the feed loads real recommendation cards. The "Trending" tab hits
 * /recommend/trending, which is corpus-global (not per-user), so its cards are
 * deterministic content — the strongest assertion that recommendations render.
 */
test('for-you renders recommendation cards after login', async ({ page }) => {
  await signUpFreshUser(page, uniqueEmail('e2e-foryou'))

  await page.goto('/for-you')
  await expect(page.getByRole('heading', { name: 'For You' })).toBeVisible()

  // Switch to the global Trending feed so the card content does not depend on a
  // per-user cold-start profile.
  await page.getByRole('button', { name: 'Trending' }).click()

  // Recommendation cards render as external article links (the for-you page
  // styles via CSS modules, so anchor to the visible link, not hashed classes).
  const cardLink = page.locator('a[href^="http"]').first()
  await expect(cardLink).toBeVisible({ timeout: 30_000 })
  // The link has non-empty visible text: the article title.
  await expect(cardLink).toHaveText(/.+/)
})
