import { expect, type Page } from '@playwright/test'

/** The scratch API port run-on-box.sh boots (see playwright.config.ts baseURL). */
export const API_BASE = process.env.E2E_API_BASE || 'http://127.0.0.1:8099'

/** Password that satisfies the backend policy (needs a letter + a digit, min 8). */
export const TEST_PASSWORD = 'E2eBoxPass1'

/** Random per-run — unique across reruns, so signup never hits a duplicate. */
export function uniqueEmail(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e9)}@example.com`
}

/**
 * Sign out through the TopBar control. The TopBar renders "Log out" only when
 * the user is authenticated (getMe resolved), so wait for it first.
 */
export async function signOut(page: Page): Promise<void> {
  await expect(page.locator('button.topbar-logout')).toBeVisible()
  await page.locator('button.topbar-logout').click()
}

/**
 * Sign up a brand-new scratch user on /signup; on success the app redirects to
 * /chat (the signup default next) and the TopBar shows the account control.
 */
export async function signUpFreshUser(page: Page, email: string): Promise<void> {
  await page.goto('/signup')
  await expect(page).toHaveURL(/\/signup/)
  await page.getByLabel(/Name/).fill('E2E User')
  await page.getByLabel('Email', { exact: true }).fill(email)
  await page.getByLabel('Password', { exact: true }).fill(TEST_PASSWORD)
  await page.getByRole('button', { name: 'Sign up' }).click()
  // Signup success redirects to /chat (default next) — proves the account stuck.
  await expect(page).toHaveURL(/\/chat$/, { timeout: 20_000 })
  await expect(page.locator('button.topbar-logout')).toBeVisible()
}

/**
 * Log in with existing credentials on /login, then land on /chat (the default)
 * and wait for the TopBar account control, which proves the session cookie works.
 */
export async function login(page: Page, email: string, password: string = TEST_PASSWORD): Promise<void> {
  await page.goto('/login')
  await page.getByLabel(/email/i).or(page.locator('input[type="email"]')).fill(email)
  await page.getByLabel(/password/i).or(page.locator('input[type="password"]')).fill(password)
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page).toHaveURL(/\/chat$/, { timeout: 20_000 })
  await expect(page.locator('button.topbar-logout')).toBeVisible()
}
