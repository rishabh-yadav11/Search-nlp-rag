import { test, expect } from '@playwright/test'
import { signUpFreshUser, uniqueEmail } from './support'

/**
 * Chat end-to-end. The scratch backend runs with NO GEMINI_API_KEY, so any turn
 * that would need the LLM must not be asked. The deterministic, zero-spend path
 * is a query with no strong corpus matches: _prepare_turn returns the honest
 * non-LLM answer ("No sufficiently relevant articles...") without ever calling
 * the model. Ask a deliberately impossible phrase so retrieval returns nothing
 * strong; the answer region must render that honest fallback.
 *
 * `/chat` is auth-gated (backend require_auth + chat:use): user role has
 * chat:use, so a freshly signed-up user can send a message.
 */
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

  // An assistant answer bubble must render — and it must be the honest fallback,
  // never a fabricated LLM claim. Wait for the message list to settle.
  const answer = page.locator('.chat-msg-bubble .chat-msg-answer').last()
  await expect(answer).toBeVisible({ timeout: 30_000 })
  await expect(answer).toContainText(new RegExp(FALLBACK_FRAGMENTS.join('|')))
})
