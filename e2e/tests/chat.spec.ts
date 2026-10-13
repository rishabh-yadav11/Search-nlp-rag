import { test, expect } from '@playwright/test'
import { signUpFreshUser, uniqueEmail } from './support'


const GIBBERISH = 'zlix poqwezm vunflorp qaaeb'

const FALLBACK_FRAGMENTS = [
  'No sufficiently relevant articles were found for this query.',
  "I couldn't find strong matches",
  "I couldn't find any articles",
]

test('chat renders the honest no-LLM fallback answer', async ({ page }) => {
  await signUpFreshUser(page, uniqueEmail('e2e-chat'))

  await page.goto('/chat')
  await expect(page.locator('textarea[aria-label="Message"]')).toBeVisible()

  await page.locator('textarea[aria-label="Message"]').fill(GIBBERISH)
  await page.getByRole('button', { name: 'Send' }).click()

  
  
  const answer = page.locator('.chat-msg-bubble .chat-msg-answer').last()
  await expect(answer).toBeVisible({ timeout: 30_000 })
  await expect(answer).toContainText(new RegExp(FALLBACK_FRAGMENTS.join('|')))
})
