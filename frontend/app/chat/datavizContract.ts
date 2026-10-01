/** Run by node in backend/tests/test_dataviz_contract.py, so it must stay import-free; every rule here mirrors app/chat.py. */

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

/** Must stay byte-identical to DATAVIZ_FENCE_PATTERN in app/chat.py; the whitespace classes are spelled out because Python's and JavaScript's `\s` differ. */
export const FENCE_SRC = '```dataviz[ \\t\\r]*\\n?([\\s\\S]*?)\\n?```[\\t\\n\\v\\f\\r ]*'
export const KINDS = ['bar', 'line', 'pie'] as const
export const VIEWS = ['table', 'bar', 'line', 'pie'] as const

export const MAX_JSON_DEPTH = 100

/** Rejects a payload holding a non-finite number anywhere (a "1e999" label survives every value check) or nesting past MAX_JSON_DEPTH. */
export function hasNonFiniteNumber(value: unknown, depth = 0): boolean {
  if (depth > MAX_JSON_DEPTH) return true
  if (typeof value === 'number') return !Number.isFinite(value)
  if (Array.isArray(value)) return value.some((v) => hasNonFiniteNumber(v, depth + 1))
  if (value !== null && typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).some((v) => hasNonFiniteNumber(v, depth + 1))
  }
  return false
}

/** `parseFloat` would NOT do: it reads a numeric prefix, so "12abc" would plot as a value the model never stated. */
export const NUMERIC_LITERAL_SRC = '[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
const NUMERIC_LITERAL = new RegExp(`^${NUMERIC_LITERAL_SRC}$`)

/** NOT `String.prototype.trim`, which also strips U+FEFF while Python's `str.strip()` leaves it. */
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
  // Object.hasOwn, not a lookup: a cell reading "constructor" must not hit Object.prototype.
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
    // An integral value_column wins even when written as a JSON float ("value_column": 1.0).
    const vc: number | null =
      typeof vcRaw === 'number' && Number.isInteger(vcRaw) && vcRaw >= 0 && vcRaw < columns.length
        ? vcRaw
        : firstNumericColumn(rowArr)
    if (vc != null && !validValueColumn(rowArr, vc)) return null
    // With no numeric column the block only renders as a text table, so a chart view is malformed.
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

/** Drops an UNCLOSED fence so the model's unfinished JSON is never rendered; twin of _strip_unclosed_fence in app/chat.py. */
export function stripOpenFence(md: string): string {
  const open = md.indexOf('```dataviz')
  if (open < 0) return md
  return md.slice(0, open)
}
