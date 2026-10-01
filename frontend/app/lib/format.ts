/** Deterministic by construction: fixed `en-US` locale and `UTC` zone, so server and browser renders cannot disagree. */

const BARE_DATE_RE = /^\d{4}-\d{2}-\d{2}$/

const MS_PER_MINUTE = 60_000
const MS_PER_HOUR = 60 * MS_PER_MINUTE
const MS_PER_DAY = 24 * MS_PER_HOUR

/** Milliseconds since the epoch, or `NaN`. A bare `YYYY-MM-DD` is a calendar date, not an instant, so it parses as LOCAL midnight — `Date.parse` would read it as UTC midnight and render a day early west of Greenwich. */
export function parseLocalDate(s: string): number {
  if (!s) return NaN
  if (BARE_DATE_RE.test(s)) {
    const [y, m, d] = s.split('-').map(Number)
    return new Date(y, m - 1, d).getTime()
  }
  return Date.parse(s)
}

export function formatDate(s: string): string {
  if (!s) return 'n/a'
  const d = new Date(parseLocalDate(s))
  if (isNaN(d.getTime())) return s
  const diff = Date.now() - d.getTime()
  if (diff < MS_PER_MINUTE) return 'just now'
  if (diff < MS_PER_HOUR) {
    const m = Math.floor(diff / MS_PER_MINUTE)
    return `${m} minute${m > 1 ? 's' : ''} ago`
  }
  if (diff < MS_PER_DAY) {
    const h = Math.floor(diff / MS_PER_HOUR)
    return `${h} hour${h > 1 ? 's' : ''} ago`
  }
  if (diff < 2 * MS_PER_DAY) return 'yesterday'
  if (diff < 7 * MS_PER_DAY) return `${Math.floor(diff / MS_PER_DAY)} days ago`
  if (diff < 30 * MS_PER_DAY) {
    const w = Math.floor(diff / (7 * MS_PER_DAY))
    return `${w} week${w > 1 ? 's' : ''} ago`
  }
  if (diff < 365 * MS_PER_DAY) {
    const m = Math.floor(diff / (30 * MS_PER_DAY))
    return `${m} month${m > 1 ? 's' : ''} ago`
  }
  const y = Math.floor(diff / (365 * MS_PER_DAY))
  return `${y} year${y > 1 ? 's' : ''} ago`
}

/** Bare dates stay pinned to UTC so the calendar day never shifts — deliberately the opposite of `parseLocalDate`. */
export function formatArticleDate(s: string): string {
  if (!s) return ''
  if (BARE_DATE_RE.test(s)) {
    const [y, m, d] = s.split('-').map(Number)
    const t = new Date(Date.UTC(y, m - 1, d)).getTime()
    if (isNaN(t)) return 'n/a'
    return new Date(t).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: 'UTC',
    })
  }
  const t = Date.parse(s)
  if (isNaN(t)) return 'n/a'
  return new Date(t).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  })
}

/** UNIX timestamp in SECONDS; passing milliseconds silently yields `just now` forever. */
export function formatEpochRelative(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`
  return formatArticleDate(new Date(ts * 1000).toISOString())
}

export function formatCost(cost: number | null | undefined): string {
  if (cost == null || Number.isNaN(Number(cost))) return '$0'
  const v = Number(cost)
  if (v === 0) return '$0'
  const maximumFractionDigits = v >= 1 ? 2 : v >= 0.01 ? 4 : 6
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits })
}

/** UNIX timestamp in SECONDS, as in `formatEpochRelative`. */
export function formatEpochDateTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
    timeZone: 'UTC',
  })
}

/** Instant in milliseconds — not the SECONDS the epoch formatters take. */
export function formatClockTime(epochMs: number): string {
  return new Date(epochMs).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'UTC',
  })
}
