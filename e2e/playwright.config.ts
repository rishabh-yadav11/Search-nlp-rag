import { defineConfig } from '@playwright/test'

/**
 * E2E config for the VCCircle hybrid-search RAG app.
 *
 * The orchestrator (run-on-box.sh) boots a fully isolated scratch stack and
 * runs `npx playwright test` with this config. The baseURL is the scratch
 * frontend, which proxies/points at the scratch API on 127.0.0.1:8099.
 *
 * run-on-box.sh exports extra vars read by the specs:
 *   E2E_SEARCH_QUERY  - a phrase guaranteed to hit the corpus (derived by the
 *                       orchestrator from the article corpus at index-build time)
 *   E2E_ADMIN_EMAIL   - bootstrap admin seeded into the scratch backend
 *   E2E_ADMIN_PASSWORD
 *
 * `fullyParallel: false` + `retries: 0` keeps the suite deterministic and
 * order-stable: at most one worker, no re-runs, so shared scratch state (the
 * auth DB, the analytics counters) is never mutated concurrently.
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000,
  expect: {
    // Search + recommend hit the cross-encoder reranker on a cold box; the
    // first query warms the model, so polling patience must be generous.
    timeout: 30_000,
  },
  use: {
    baseURL: 'http://127.0.0.1:3099',
    trace: 'on-first-retry',
  },
  reporter: [['list']],
})
