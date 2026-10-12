import { test, expect } from '@playwright/test'

/**
 * SSR end-to-end smoke: GET / must return the rendered app shell. This proves
 * the full chain — Next build/start -> the page hydrate -> the API base baked
 * into the build. No API call is needed here; the shell itself is the assertion.
 */
test('GET / renders the app shell with the search UI', async ({ page }) => {
  const response = await page.goto('/')

  expect(response?.status()).toBe(200)

  // App shell: the VCCircle wordmark in the TopBar. The footer renders a second
  // VCCircle logo, so the locator must be narrowed to the banner one.
  await expect(page.locator('header').getByRole('img', { name: 'VCCircle' })).toBeVisible()

  // The home + search entry point: a search input and its submit button.
  await expect(page.getByLabel('Search query')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Search', exact: true })).toBeVisible()

  // Suggestion chips render too (they feed the same search handler).
  await expect(page.getByRole('group', { name: 'Search suggestions' })).toBeVisible()
})

test('mobile 375px has no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await page.goto('/')

  const overflow = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    inner: window.innerWidth,
    topbarNavRight: document.querySelector('.topbar-nav')?.getBoundingClientRect().right,
  }))
  expect(overflow.scroll).toBeLessThanOrEqual(overflow.inner)
  expect(overflow.topbarNavRight!).toBeLessThanOrEqual(overflow.inner)
})
