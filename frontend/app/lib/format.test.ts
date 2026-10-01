import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  formatArticleDate,
  formatClockTime,
  formatCost,
  formatDate,
  formatEpochDateTime,
  formatEpochRelative,
  parseLocalDate,
} from './format'

/**
 * The stored `"YYYY-MM-DD"` string is a calendar date, not a UTC instant: handing it to `new Date(...)`
 * reads it as UTC midnight and renders it in the viewer's zone, so the same article showed a different
 * day in a card than on the search page west of Greenwich.
 */

type Fn = (s: string) => string

describe('parseLocalDate — a bare YYYY-MM-DD is a local calendar date', () => {
  it('parses a bare date at local midnight, offset from UTC by the local zone', () => {
    // A local Date constructor is what makes this a local calendar date; `Date.parse('2024-05-01')`
    // yields UTC midnight instead. Stating the invariant as the offset holds on a UTC runner too, and
    // `getTimezoneOffset()` (minutes west of UTC) carries the sign directly: negating it would invert
    // the comparison and make UTC yield -0, which `toBe`'s Object.is check rejects against +0.
    const localMidnight = new Date(2024, 4, 1).getTime()
    const utcMidnight = Date.parse('2024-05-01')
    expect(parseLocalDate('2024-05-01')).toBe(localMidnight)
    expect(parseLocalDate('2024-05-01') - utcMidnight).toBe(
      new Date(2024, 4, 1).getTimezoneOffset() * 60_000,
    )
  })

  it('reads a full timestamp as the instant it is', () => {
    expect(parseLocalDate('2024-05-01T10:30:00Z')).toBe(Date.parse('2024-05-01T10:30:00Z'))
  })

  it('yields NaN for an unusable value', () => {
    expect(parseLocalDate('')).toBeNaN()
    expect(parseLocalDate('not a date')).toBeNaN()
  })
})

describe('formatArticleDate — the same day for every viewer', () => {
  const FORMATS: Fn[] = [formatArticleDate]

  it('renders a bare stored date as that exact calendar day', () => {
    // The point of the consolidation: the label is the day in the record, not a viewer-dependent day.
    for (const format of FORMATS) {
      expect(format('2024-05-01')).toBe('May 1, 2024')
      expect(format('2024-12-31')).toBe('Dec 31, 2024')
      expect(format('2024-01-01')).toBe('Jan 1, 2024')
    }
  })

  it('renders a full timestamp at its UTC day', () => {
    for (const format of FORMATS) {
      expect(format('2024-05-01T00:30:00Z')).toBe('May 1, 2024')
    }
  })

  it('does not roll a bare date to the previous day', () => {
    // The concrete failure the cards had: UTC midnight in a timezone behind UTC lands on the day before.
    for (const format of FORMATS) {
      expect(format('2024-01-01')).not.toMatch(/Dec 31, 2023/)
    }
  })

  it('returns empty for a missing date and a marker for a non-date', () => {
    for (const format of FORMATS) {
      expect(format('')).toBe('')
      expect(format('sometime last spring')).toBe('n/a')
    }
  })
})

describe('formatDate — relative labels', () => {
  const NOW = new Date('2024-05-10T12:00:00Z').getTime()
  const AT = (iso: string) => iso

  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('labels each magnitude, and a bare date relative to local midnight', () => {
    expect(formatDate(AT('2024-05-10T11:59:30Z'))).toBe('just now')
    expect(formatDate(AT('2024-05-10T11:30:00Z'))).toBe('30 minutes ago')
    expect(formatDate(AT('2024-05-10T11:00:00Z'))).toBe('1 hour ago')
    expect(formatDate(AT('2024-05-10T09:00:00Z'))).toBe('3 hours ago')
    expect(formatDate(AT('2024-05-09T12:00:00Z'))).toBe('yesterday')
    expect(formatDate(AT('2024-05-07T12:00:00Z'))).toBe('3 days ago')
    expect(formatDate(AT('2024-04-20T12:00:00Z'))).toBe('2 weeks ago')
    expect(formatDate(AT('2024-02-10T12:00:00Z'))).toBe('3 months ago')
    expect(formatDate(AT('2021-05-10T12:00:00Z'))).toBe('3 years ago')
  })

  it('is singular for one unit and plural otherwise', () => {
    expect(formatDate(AT('2024-05-10T11:59:00Z'))).toBe('1 minute ago')
    expect(formatDate(AT('2024-04-27T12:00:00Z'))).toBe('1 week ago')
  })

  it('reports a missing or unusable value without rendering a blank', () => {
    expect(formatDate('')).toBe('n/a')
    expect(formatDate('sometime last spring')).toBe('sometime last spring')
  })
})

describe('formatEpochRelative — labels a UNIX timestamp in seconds', () => {
  const NOW = new Date('2024-05-10T12:00:00Z').getTime()
  const T = Math.floor(NOW / 1000)

  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('labels recent timestamps in short units', () => {
    expect(formatEpochRelative(T - 5)).toBe('just now')
    expect(formatEpochRelative(T - 120)).toBe('2m ago')
    expect(formatEpochRelative(T - 7200)).toBe('2h ago')
    expect(formatEpochRelative(T - 3 * 86400)).toBe('3d ago')
  })

  it('falls back to the same absolute label the article cards use', () => {
    const old = Math.floor(Date.UTC(2024, 4, 1) / 1000)
    expect(formatEpochRelative(old)).toBe('May 1, 2024')
  })
})

describe('formatEpochDateTime — a pinned absolute instant', () => {
  it('renders a known timestamp to a known string', () => {
    // A literal, not a re-derivation: the dashboard test delegates to this function, so re-deriving
    // here would compare the page against itself.
    expect(formatEpochDateTime(1_700_000_000)).toBe('Nov 14, 2023, 10:13 PM')
  })

  it('does not shift with the ambient timezone', () => {
    // 10:13 PM UTC is 02:13 the next day in Los Angeles, so the `timeZone: 'UTC'` pin is asserted, not assumed.
    const previousTz = process.env.TZ
    try {
      process.env.TZ = 'America/Los_Angeles'
      expect(formatEpochDateTime(1_700_000_000)).toBe('Nov 14, 2023, 10:13 PM')
    } finally {
      process.env.TZ = previousTz
    }
  })
})

describe('formatClockTime — a pinned clock label', () => {
  it('renders a known instant to a known string', () => {
    expect(formatClockTime(1_700_000_000_000)).toBe('10:13 PM')
  })

  it('does not shift with the ambient timezone', () => {
    const previousTz = process.env.TZ
    try {
      process.env.TZ = 'Asia/Tokyo'
      expect(formatClockTime(1_700_000_000_000)).toBe('10:13 PM')
    } finally {
      process.env.TZ = previousTz
    }
  })
})

describe('formatCost — one US-dollar format for every surface', () => {
  it('scales the fraction to the magnitude and pins the grouping', () => {
    // At or above a dollar: whole cents.
    expect(formatCost(1)).toBe('$1.00')
    expect(formatCost(1.5)).toBe('$1.50')
    expect(formatCost(12.3456)).toBe('$12.35')
    expect(formatCost(1234.5)).toBe('$1,234.50')
    // Below a dollar: enough sub-cent precision for a real per-message cost, never padded with
    // meaningless zeros, and at least two decimal places so it still reads as money.
    expect(formatCost(0.5)).toBe('$0.50')
    expect(formatCost(0.0123)).toBe('$0.0123')
    expect(formatCost(0.000001)).toBe('$0.000001')
  })

  it('never shows a fraction finer than the amount carries', () => {
    // A whole-dollar amount is shown as dollars and cents, not padded to six decimals.
    expect(formatCost(99.999)).toBe('$100.00')
    expect(formatCost(1234.5678)).toBe('$1,234.57')
  })

  it('shows a zero rather than a blank for an absent or zero cost', () => {
    expect(formatCost(0)).toBe('$0')
    expect(formatCost(null)).toBe('$0')
    expect(formatCost(undefined)).toBe('$0')
    expect(formatCost(NaN)).toBe('$0')
  })
})
