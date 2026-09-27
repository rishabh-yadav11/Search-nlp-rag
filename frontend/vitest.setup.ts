import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// `globals` is off, so Testing Library's automatic cleanup never registers.
afterEach(() => {
  cleanup()
})
