import { test, expect } from '@playwright/test'


test('GET / renders the app shell with the search UI', async ({ page }) => {
  const response = await page.goto('/')

  expect(response?.status()).toBe(200)

  
  
  await expect(page.locator('header').getByRole('img', { name: 'VCCircle' })).toBeVisible()

  
  await expect(page.getByLabel('Search query')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Search', exact: true })).toBeVisible()

  
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
