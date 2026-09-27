/**
 * The single place the frontend formats dates and costs.
 *
 * Every page and card used to carry its own copy, which is how the same
 * article came to show one day on the search page and another in a
 * recommendation card: the search page parsed the stored `"YYYY-MM-DD…"`
 * string as a LOCAL calendar date (correct — a bare date is not a UTC
 * instant), the cards handed it straight to `new Date(...)`, which treats it as
 * UTC midnight and then rendered it in the viewer's zone, so any viewer
 * west of Greenwich saw the previous day. It also meant locale- and
 * timezone-dependent output in a client-rendered string.
 *
 * Everything here is therefore deterministic: fixed `en-US` locale, fixed
 * `UTC` where a zone is involved, and one shared parse.
 */

/** A bare calendar date as the backend stores it, e.g. `2024-05-01`. */
const BARE_DATE_RE = /^\d{4}-\d{2}-\d{2}$/

const MS_PER_MINUTE = 60_000
const MS_PER_HOUR = 60 * MS_PER_MINUTE
const MS_PER_DAY = 24 * MS_PER_HOUR

/**
 * Milliseconds since the epoch for a stored date string, or `NaN`.
 *
 * A bare `YYYY-MM-DD` is a calendar date, not an instant: `Date.parse` reads it
 * as UTC midnight, which renders as the previous day for anyone west of
 * Greenwich. It is parsed as LOCAL midnight instead. Anything else (a full
 * ISO timestamp, which really is an instant) goes to `Date.parse` unchanged.
 */
export function parseLocalDate(s: string): number {
  if (!s) return NaN
  if (BARE_DATE_RE.test(s)) {
    const [y, m, d] = s.split('-').map(Number)
    return new Date(y, m - 1, d).getTime()
  }
  return Date.parse(s)
}

/**
 * A human relative label for a stored date string ("just now", "3 days ago",
 * "2 years ago"). Returns `'n/a'` for an empty value and the raw string when
 * it cannot be parsed, so an unexpected backend value stays visible instead of
 * rendering as a blank.
 */
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

/**
 * A stable absolute calendar date for an article card ("May 1, 2024").
 *
 * Fixed `en-US` locale and fixed `UTC` zone, so the same record renders
 * identically for every viewer, in the server and the browser, and cannot
 * disagree with the relative label `formatDate` derives from the same parse.
 * Returns `''` for an empty value (callers already hide the slot) and
 * `'n/a'` for a value that is not a date at all.
 */
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

/**
 * A relative label for a UNIX timestamp in SECONDS ("3m ago"), falling back to
 * an absolute date past a week. Fixed locale and zone for the same reason as
 * `formatArticleDate`; `now` is read at call time so the label keeps advancing
 * while the page is open.
 */
export function formatEpochRelative(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`
  return formatArticleDate(new Date(ts * 1000).toISOString())
}

/**
 * A US-dollar amount, with the fraction sized to the magnitude: whole cents at
 * or above $1, enough sub-cent precision below it to show a real per-message
 * cost, and never padded out with meaningless zeros. This is the rule the chat
 * usage line already approximated with a magnitude branch, and the analytics
 * cards already approximated with a fixed 2-to-6 range; the shared rule keeps
 * the first behaviour for chat's sub-dollar costs and the second for the
 * dashboard's large totals.
 *
 * Grouping and fraction are locale-pinned to `en-US`, which is the whole
 * reason this lives in one module rather than at each call site.
 */
export function formatCost(cost: number | null | undefined): string {
  if (cost == null || Number.isNaN(Number(cost))) return '$0'
  const v = Number(cost)
  if (v === 0) return '$0'
  const maximumFractionDigits = v >= 1 ? 2 : v >= 0.01 ? 4 : 6
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits })
}

/**
 * A full date-and-time label for a UNIX timestamp in seconds, for the places
 * that show when something happened rather than how long ago ("Nov 14, 2023,
 * 10:13 PM"). Fixed locale and zone for the same reason as `formatArticleDate`.
 *
 * This is the shared replacement for the analytics dashboard's inline
 * `toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })`, whose bare
 * `[]` locale let the server's rendering and the browser's disagree.
 */
export function formatEpochDateTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
    timeZone: 'UTC',
  })
}
