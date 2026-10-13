import { expect, type Page } from '@playwright/test'


export const API_BASE = process.env.E2E_API_BASE || 'http://127.0.0.1:8099'


export const TEST_PASSWORD = 'E2eBoxPass1'


export function uniqueEmail(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e9)}@example.com`
}


export async function signOut(page: Page): Promise<void> {
  await expect(page.locator('button.topbar-logout')).toBeVisible()
  await page.locator('button.topbar-logout').click()
}


export async function signUpFreshUser(page: Page, email: string): Promise<void> {
  await page.goto('/signup')
  await expect(page).toHaveURL(/\/signup/)
  await page.getByLabel(/Name/).fill('E2E User')
  await page.getByLabel('Email', { exact: true }).fill(email)
  await page.getByLabel('Password', { exact: true }).fill(TEST_PASSWORD)
  await page.getByRole('button', { name: 'Sign up' }).click()
  
  
  await expect(page.getByRole('heading', { name: 'Almost there' })).toBeVisible({ timeout: 20_000 })
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page).toHaveURL(/\/login/)
  await login(page, email)
}


export async function login(page: Page, email: string, password: string = TEST_PASSWORD): Promise<void> {
  await page.goto('/login')
  await page.getByLabel(/email/i).or(page.locator('input[type="email"]')).fill(email)
  await page.getByLabel(/password/i).or(page.locator('input[type="password"]')).fill(password)
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page).toHaveURL(/\/chat$/, { timeout: 20_000 })
  await expect(page.locator('button.topbar-logout')).toBeVisible()
}
