import { test, expect } from '@playwright/test'


const QUERY = process.env.E2E_SEARCH_QUERY || 'Ola Electric IPO'

test('search returns real result cards and a link stays on-site', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('Search query').fill(QUERY)
  await page.getByRole('button', { name: 'Search', exact: true }).click()

  
  await expect(page.locator('.results-heading')).toContainText(QUERY)
  const firstResult = page.locator('div.result').first()
  await expect(firstResult).toBeVisible()

  
  
  const link = firstResult.locator('a[href^="http"]').first()
  await expect(link).toBeVisible()
  const href = await link.getAttribute('href')
  expect(href).toBeTruthy()
  expect(new URL(href!).hostname.length).toBeGreaterThan(0)

  const [popup] = await Promise.all([page.waitForEvent('popup'), link.click()])
  expect(new URL(popup.url()).hostname).not.toBe(new URL(page.url()).hostname)
  await popup.close()

  
  await expect(page).toHaveURL(/127\.0\.0\.1:3099/)
  await expect(page.getByLabel('Search query')).toBeVisible()
})
