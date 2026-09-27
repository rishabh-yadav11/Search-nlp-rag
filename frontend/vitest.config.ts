import { defineConfig } from 'vitest/config'

export default defineConfig({
  // Next's tsconfig sets `jsx: "preserve"` (SWC compiles it), so esbuild has to
  // be told to use the automatic runtime or JSX compiles to `React.createElement`
  // against an undefined `React`.
  esbuild: { jsx: 'automatic', jsxImportSource: 'react' },
  test: {
    environment: 'jsdom',
    include: ['app/**/*.test.{ts,tsx}'],
    setupFiles: ['./vitest.setup.ts'],
  },
})
