import { test, expect } from '@playwright/test'

/**
 * The corpus-derived query is exported by run-on-box.sh. Be explicit about
 * falling back to a known-good corpus topic when unset (e.g. running locally
 * against a prebuilt stack) — never an empty string, which would change the
 * request shape and mask a config bug with a fake failure.
 */
const QUERY = process.env.E2E_SEARCH_QUERY || 'Ola Electric IPO'

test('search returns real result cards and a link stays on-site', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('Search query').fill(QUERY)
  await page.getByRole('button', { name: 'Search', exact: true }).click()

  // A results heading ("<n> results for "<query>"") and at least one card.
  await expect(page.locator('.results-heading')).toContainText(QUERY)
  const firstResult = page.locator('div.result').first()
  await expect(firstResult).toBeVisible()

  // Card links are the real article URLs (external host), opened in a new tab
  // (target=_blank): the app itself must stay on-site.
  const link = firstResult.locator('a[href^="http"]').first()
  await expect(link).toBeVisible()
  const href = await link.getAttribute('href')
  expect(href).toBeTruthy()
  expect(new URL(href!).hostname.length).toBeGreaterThan(0)

  const [popup] = await Promise.all([page.waitForEvent('popup'), link.click()])
  expect(new URL(popup.url()).hostname).not.toBe(new URL(page.url()).hostname)
  await popup.close()

  // App still on the scratch origin — the click did not navigate away.
  await expect(page).toHaveURL(/127\.0\.0\.1:3099/)
  await expect(page.getByLabel('Search query')).toBeVisible()
})
