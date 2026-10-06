/**
 * The ONE dataviz block grammar and validator, shared by the renderer
 * (`DataViz.tsx`) and the backend (`parse_dataviz` in backend/app/chat.py).
 *
 * This module holds the whole accept/reject decision — fence regex, missing
 * value tokens, numeric coercion, value-column selection, label check — and
 * NOTHING else: no React, no JSX, no imports. That keeps it the single place
 * the browser decides a block is renderable, mirroring backend/app/chat.py so
 * the two validators agree. Nothing checks that they do any more, so a change
 * to one rule has to be made in the other by hand.
 */

// Types

export type DataVizBlock = {
  title?: string
  columns: string[]
  rows: (string | number)[][]
  value_column: number | null
  format?: string
  kind?: 'bar' | 'line' | 'pie'
  view?: 'table' | 'bar' | 'line' | 'pie'
}

export type ContentPart =
  | { type: 'md'; md: string }
  | { type: 'viz'; block: DataVizBlock }
  | { type: 'err' }

// Constants

/**
 * Fence grammar, character-for-character identical to DATAVIZ_FENCE_PATTERN in
 * backend/app/chat.py; the two source strings must stay equal. The newline
 * after the tag is OPTIONAL, so a fence written as
 * ```dataviz{...}``` is the same grammar.
 *
 * Whitespace is spelled as an explicit ASCII class, NEVER `\s` or `[^\S\n]`:
 * Python's and JavaScript's `\s`/`\S` cover DIFFERENT Unicode whitespace
 * (U+FEFF, U+0085, U+001C-U+001F), so the two engines would consume different
 * characters — a fence carrying a BOM was kept by the server and dropped by
 * the browser — while the pattern strings still compared equal. `[ \t\r]*`
 * says the same thing in both engines; `\r` is there for CRLF answers.
 *
 * Because the trailing class also changes how much trailing text a side
 * consumes rather than the verdict, the two implementations must be compared on
 * the full match SPAN, not just on the accept/reject outcome.
 */
export const FENCE_SRC = '```dataviz[ \\t\\r]*\\n?([\\s\\S]*?)\\n?```[\\t\\n\\v\\f\\r ]*'
export const KINDS = ['bar', 'line', 'pie'] as const
export const VIEWS = ['table', 'bar', 'line', 'pie'] as const

// Helpers

export const MAX_JSON_DEPTH = 100

/**
 * True when ANY number in the payload is not a finite double, or when the
 * payload nests deeper than MAX_JSON_DEPTH.
 *
 * Anywhere, not only in the value column: a LABEL cell reading 1e999 survives
 * every value check but can never be displayed, and an integer literal too
 * wide for Python's int conversion makes the backend's json.loads reject the
 * whole payload while JSON.parse quietly yields Infinity. The backend applies
 * the identical walk (_has_non_finite in backend/app/chat.py, same depth
 * limit), so both sides reject the same blocks.
 */
export function hasNonFiniteNumber(value: unknown, depth = 0): boolean {
  if (depth > MAX_JSON_DEPTH) return true
  if (typeof value === 'number') return !Number.isFinite(value)
  if (Array.isArray(value)) return value.some((v) => hasNonFiniteNumber(v, depth + 1))
  if (value !== null && typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).some((v) => hasNonFiniteNumber(v, depth + 1))
  }
  return false
}

/**
 * A cell counts as a number only when it IS one, in full.
 *
 * Bools are rejected first, because the backend must test `isinstance(v, bool)`
 * explicitly. A NUMBER is accepted only when finite, so a JSON `NaN`/
 * `Infinity` literal can never reach a chart's min/max.
 *
 * A STRING counts only when the whole comma-stripped, trimmed cell is a plain
 * numeric literal. `parseFloat` would NOT do: it reads a numeric PREFIX, so
 * "12abc" would plot as 12 — a value the model never stated, and one the
 * backend's `float()` accepts. NUMERIC_LITERAL_SRC is the character-for-
 * character twin of _NUMERIC_LITERAL_SRC in backend/app/chat.py ([0-9] rather
 * than \d, so it means the same in both languages) and the two must stay equal:
 * both sides accept the same spellings ("1,200",
 * " 1.5 ", "+3", "1e3") and reject the same impostors ("12abc", "0x10",
 * "1_000", "inf", "nan", "").
 */
export const NUMERIC_LITERAL_SRC = '[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
const NUMERIC_LITERAL = new RegExp(`^${NUMERIC_LITERAL_SRC}$`)

/**
 * The whitespace trimmed off a cell, spelled out as ASCII whitespace. NOT
 * `String.prototype.trim`, which also removes U+FEFF while Python's
 * `str.strip()` leaves it: a cell carrying a BOM was "missing" here and a real
 * value on the server. Each of these characters is probed against the backend's
 * `_TRIM_CHARS` by hand.
 */
export const TRIM_SRC = '\\t\\n\\v\\f\\r '

const TRIM = new RegExp(`^[${TRIM_SRC}]+|[${TRIM_SRC}]+$`, 'g')

export const trimCell = (s: string): string => s.replace(TRIM, '')

export function toNum(v: unknown): number | null {
  if (typeof v === 'boolean') return null
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string') {
    const cleaned = trimCell(v.replace(/,/g, ''))
    if (!NUMERIC_LITERAL.test(cleaned)) return null
    const n = Number(cleaned)
    return Number.isFinite(n) ? n : null
  }
  return null
}

/**
 * Cells that mean "no value stated here" rather than a number, so a top-N
 * table can list an item whose value the articles never gave. The key set must
 * stay equal to the backend's _MISSING_VALUE_TOKENS and to every other copy of
 * it; a one-sided addition silently changes what the UI calls "not stated".
 */
export const MISSING_VALUE_TOKENS: Record<string, true> = {
  '': true,
  'value not stated': true,
  'not stated': true,
  'n/a': true,
  na: true,
  'n/d': true,
  nil: true,
  none: true,
  unknown: true,
  tbd: true,
  'to be decided': true,
  'to be determined': true,
  '—': true,
  '-': true,
  '--': true,
}

export function isMissing(v: unknown): boolean {
  if (v == null) return true
  // Object.hasOwn, not a plain lookup: a cell reading "constructor" or
  // "__proto__" must not find Object.prototype's members and count as missing.
  if (typeof v === 'string') {
    return Object.hasOwn(MISSING_VALUE_TOKENS, trimCell(v).toLowerCase())
  }
  return false
}

export function validValueColumn(rows: (string | number)[][], j: number): boolean {
  const present = rows.map((r) => r[j]).filter((v) => !isMissing(v))
  return present.length > 0 && present.every((v) => toNum(v) != null)
}

export function firstNumericColumn(rows: (string | number)[][]): number | null {
  if (!rows.length || !rows[0].length) return null
  for (let j = 0; j < rows[0].length; j++) {
    if (validValueColumn(rows, j)) return j
  }
  return null
}

export function hasLabelContent(rows: (string | number)[][], columns: string[], valueColumn: number | null): boolean {
  const labelCols = columns.map((_, j) => j).filter((j) => j !== valueColumn)
  if (!labelCols.length) return true
  return labelCols.some((j) => rows.some((r) => !isMissing(r[j])))
}

export function parseDataViz(text: string): DataVizBlock | null {
  const m = new RegExp(FENCE_SRC).exec(text)
  if (!m) return null
  try {
    const d = JSON.parse(m[1])
    if (!d || typeof d !== 'object') return null
    if (hasNonFiniteNumber(d)) return null
    const columns: unknown = (d as { columns?: unknown }).columns
    const rows: unknown = (d as { rows?: unknown }).rows
    if (!Array.isArray(columns) || !columns.length || !columns.every((c) => typeof c === 'string')) return null
    if (!Array.isArray(rows) || !rows.length || !rows.every((r) => Array.isArray(r))) return null
    const rowArr = rows as (string | number)[][]
    if (rowArr.some((r) => r.length !== columns.length)) return null
    const vcRaw = (d as { value_column?: unknown }).value_column
    const kind = (d as { kind?: unknown }).kind
    const view = (d as { view?: unknown }).view
    const viewName =
      typeof view === 'string' && (VIEWS as readonly string[]).includes(view)
        ? (view as DataVizBlock['view'])
        : undefined
    // An explicit integral value_column wins, including a whole number written
    // as a JSON float ("value_column": 1.0): rejecting it and silently re-picking
    // the first numeric column would plot one series while the server had
    // validated another. Everything else (missing, bool, string, fractional, out
    // of range) falls back to the first numeric column, exactly as the backend does.
    const vc: number | null =
      typeof vcRaw === 'number' && Number.isInteger(vcRaw) && vcRaw >= 0 && vcRaw < columns.length
        ? vcRaw
        : firstNumericColumn(rowArr)
    if (vc != null && !validValueColumn(rowArr, vc)) return null
    // A block with no numeric column is only renderable as a plain text table;
    // chart views have nothing to plot, so they are malformed and the server
    // strips them. This mirrors the backend's `vc is None and view != "table"`
    // rejection, so the browser does not hand a null value_column to the renderer.
    if (vc == null && viewName !== 'table') return null
    if (!hasLabelContent(rowArr, columns, vc)) return null
    return {
      title: typeof (d as { title?: unknown }).title === 'string' ? (d as { title: string }).title : undefined,
      columns: columns as string[],
      rows: rowArr,
      value_column: vc,
      format: typeof (d as { format?: unknown }).format === 'string' ? (d as { format: string }).format : undefined,
      kind: typeof kind === 'string' && (KINDS as readonly string[]).includes(kind) ? (kind as DataVizBlock['kind']) : undefined,
      view: viewName,
    }
  } catch {
    return null
  }
}

export function splitContent(text: string): ContentPart[] {
  const re = new RegExp(FENCE_SRC, 'g')
  const parts: ContentPart[] = []
  let last = 0
  let m: RegExpExecArray | null
  let found = false
  while ((m = re.exec(text))) {
    found = true
    if (m.index > last) parts.push({ type: 'md', md: text.slice(last, m.index) })
    const block = parseDataViz(m[0])
    if (block) parts.push({ type: 'viz', block })
    else parts.push({ type: 'err' })
    last = re.lastIndex
  }
  if (!found) return [{ type: 'md', md: stripOpenFence(text) }]
  if (last < text.length) parts.push({ type: 'md', md: text.slice(last) })
  return parts.map((p) =>
    p.type === 'md' ? { type: 'md' as const, md: stripOpenFence(p.md) } : p,
  )
}

/**
 * Truncate an UNCLOSED dataviz fence: everything from its opening marker to
 * the end of the markdown chunk is dropped, so the raw JSON behind a fence the
 * model never finished is never rendered as text.
 *
 * The frontend half of the ONE fence rule shared with the backend
 * (`_strip_unclosed_fence`, compiled from DATAVIZ_FENCE_PATTERN in
 * backend/app/chat.py): both truncate from the marker to the end, so the stored
 * answer and the rendered answer never disagree.
 *
 * A closed, valid fence never reaches here — `splitContent` consumes those — so
 * any marker left in a markdown chunk starts an unclosed block. Exported so the
 * backend can check that the two implementations truncate at the same offset on
 * the same text.
 */
export function stripOpenFence(md: string): string {
  const open = md.indexOf('```dataviz')
  if (open < 0) return md
  return md.slice(0, open)
}
