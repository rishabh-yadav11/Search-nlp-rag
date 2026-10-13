import { defineConfig } from '@playwright/test'


export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000,
  expect: {
    
    
    timeout: 30_000,
  },
  use: {
    baseURL: 'http://127.0.0.1:3099',
    trace: 'on-first-retry',
  },
  reporter: [['list']],
})
