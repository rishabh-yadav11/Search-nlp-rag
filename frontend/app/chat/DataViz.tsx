'use client'

import { useEffect, useMemo, useState } from 'react'

export type DataVizBlock = {
  title?: string
  columns: string[]
  rows: (string | number)[][]
  value_column: number | null
  format?: string
  kind?: 'bar' | 'line' | 'pie'
  view?: 'table' | 'bar' | 'line' | 'pie' | 'picto'
}

type ContentPart = { type: 'md'; md: string } | { type: 'viz'; block: DataVizBlock }

const FENCE_SRC = '```dataviz\\s*\\n([\\s\\S]*?)\\n```'
const KINDS = ['bar', 'line', 'pie'] as const
const VIEWS = ['table', 'bar', 'line', 'pie', 'picto'] as const

function toNum(v: unknown): number | null {
  if (typeof v === 'boolean') return null
  if (typeof v === 'number') return v
  if (typeof v === 'string') {
    const n = parseFloat(v.replace(/,/g, ''))
    return Number.isFinite(n) ? n : null
  }
  return null
}

const MISSING_VALUE_TOKENS = new Set([
  '', 'value not stated', 'not stated', 'n/a', 'na', 'n/d', 'nil', 'none',
  'unknown', 'tbd', 'to be decided', 'to be determined', '—', '-', '--',
])

function isMissing(v: unknown): boolean {
  if (v == null) return true
  if (typeof v === 'string') return MISSING_VALUE_TOKENS.has(v.trim().toLowerCase())
  return false
}

// A value column must hold a number in every non-missing cell and at least one
// number overall (empty cells are allowed, e.g. 'value not stated').
function validValueColumn(rows: (string | number)[][], j: number): boolean {
  const present = rows.map((r) => r[j]).filter((v) => !isMissing(v))
  return present.length > 0 && present.every((v) => toNum(v) != null)
}

function firstNumericColumn(rows: (string | number)[][]): number | null {
  if (!rows.length || !rows[0].length) return null
  for (let j = 0; j < rows[0].length; j++) {
    if (validValueColumn(rows, j)) return j
  }
  return null
}

// A data block is only useful if at least one non-value column carries
// identifying text. A table whose label cells are all empty shows only numbers
// and is treated as malformed (mirrors backend app.chat._has_label_content), so
// the streamed view drops it instead of rendering a column of blank names.
function hasLabelContent(rows: (string | number)[][], columns: string[], valueColumn: number | null): boolean {
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
    const columns: unknown = (d as { columns?: unknown }).columns
    const rows: unknown = (d as { rows?: unknown }).rows
    if (!Array.isArray(columns) || !columns.length || !columns.every((c) => typeof c === 'string')) return null
    if (!Array.isArray(rows) || !rows.length || !rows.every((r) => Array.isArray(r))) return null
    const rowArr = rows as (string | number)[][]
    if (rowArr.some((r) => r.length !== columns.length)) return null
    const vcRaw = (d as { value_column?: unknown }).value_column
    const vc: number | null =
      typeof vcRaw === 'number' && Number.isInteger(vcRaw) && vcRaw >= 0 && vcRaw < columns.length
        ? vcRaw
        : firstNumericColumn(rowArr)
    if (vc != null && !validValueColumn(rowArr, vc)) return null
    if (!hasLabelContent(rowArr, columns, vc)) return null
    const kind = (d as { kind?: unknown }).kind
    const view = (d as { view?: unknown }).view
    return {
      title: typeof (d as { title?: unknown }).title === 'string' ? (d as { title: string }).title : undefined,
      columns: columns as string[],
      rows: rowArr,
      value_column: vc as number,
      format: typeof (d as { format?: unknown }).format === 'string' ? (d as { format: string }).format : undefined,
      kind: typeof kind === 'string' && (KINDS as readonly string[]).includes(kind) ? (kind as DataVizBlock['kind']) : undefined,
      view: typeof view === 'string' && (VIEWS as readonly string[]).includes(view) ? (view as DataVizBlock['view']) : undefined,
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
    last = re.lastIndex
  }
  if (!found) return [{ type: 'md', md: text }]
  if (last < text.length) parts.push({ type: 'md', md: text.slice(last) })
  // If the fence is still open mid-stream, drop the trailing raw JSON so it
  // doesn't flash as a code block before the closing fence arrives.
  return parts.map((p) =>
    p.type === 'md' ? { type: 'md' as const, md: stripOpenFence(p.md) } : p,
  )
}

function stripOpenFence(md: string): string {
  const open = md.indexOf('```dataviz')
  if (open < 0) return md
  const close = md.indexOf('```', open + 3)
  if (close >= 0) return md
  return md.slice(0, open)
}

function formatValue(v: number, format?: string): string {
  switch (format) {
    case '%':
      return `${Math.round(v * 100) / 100}%`
    case '$B':
      return `$${trim(v)}B`
    case '$M':
      return `$${trim(v)}M`
    case '₹ Cr':
      return `₹${trim(v)} Cr`
    case '₹B':
      return `₹${trim(v)}B`
    default:
      return v.toLocaleString('en-US', { maximumFractionDigits: 2 })
  }
}

function trim(v: number): string {
  const s = Math.round(v * 100) / 100
  return Number.isInteger(s) ? String(s) : String(s)
}

const COLORS = ['#1a5fb4', '#26a269', '#e66100', '#c01c28', '#613583', '#2a7bde', '#d67600', '#8f5902', '#4e9a06', '#75507b']

function BarChart({ block }: { block: DataVizBlock }) {
  const { rows, columns, value_column: vc, format } = block
  const v = vc ?? 0
  const values = rows.map((r) => toNum(r[v]) ?? 0)
  // Domain spans the actual min/max (anchored at 0) so negatives render below
  // the zero baseline instead of being clipped or drawn the wrong way.
  const maxVal = Math.max(0, ...values)
  const minVal = Math.min(0, ...values)
  const n = rows.length
  const slot = 64
  const padL = 56
  const width = Math.max(360, n * slot + padL + 20)
  const height = 244
  const plotTop = 16
  const plotBottom = height - 62
  const plotH = plotBottom - plotTop
  const barW = Math.min(44, slot - 16)
  // Rotate labels only when there are enough bars that horizontal ones would
  // collide; the shallow angle + centered anchor keeps every name inside the
  // viewBox even at the plot edges.
  const rotate = n > 6
  const labelY = height - 34
  const angle = rotate ? -20 : 0

  // Map a value to a y pixel; the scale accounts for both positive and
  // negative extents so the zero line sits wherever the data requires.
  const domain = maxVal - minVal || 1
  const yOf = (val: number) => plotBottom - ((val - minVal) / domain) * plotH
  const zeroY = yOf(0)

  return (
    <div className="chat-viz-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={block.title || 'bar chart'}>
        {/* Zero baseline (only meaningful when data crosses zero). */}
        {minVal < 0 && (
          <line x1={padL - 4} y1={zeroY} x2={width - 12} y2={zeroY} stroke="#9a9a9a" strokeWidth={1} strokeDasharray="3 3" />
        )}
        {values.map((val, i) => {
          // Bars grow from the zero baseline: upward for positives, downward
          // for negatives.
          const barTop = val >= 0 ? yOf(val) : zeroY
          const bh = Math.max(2, Math.abs(yOf(val) - zeroY))
          const x = padL + i * slot
          const y = barTop
          const valTextY = val >= 0 ? barTop - 5 : barTop + bh + 12
          const full = String(rows[i][0] ?? '')
          const label = full.length > 14 ? `${full.slice(0, 13)}…` : full
          return (
            <g key={i}>
              <title>{full}</title>
              <rect x={x} y={y} width={barW} height={bh} rx={3} fill={COLORS[i % COLORS.length]} />
              <text x={x + barW / 2} y={valTextY} textAnchor="middle" className="chat-viz-bar-val">
                {formatValue(val, format)}
              </text>
              <text
                x={x + barW / 2}
                y={labelY}
                textAnchor="middle"
                transform={`rotate(${angle} ${x + barW / 2} ${labelY})`}
                className="chat-viz-bar-label"
              >
                {label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function LineChart({ block }: { block: DataVizBlock }) {
  const { rows, columns, value_column: vc, format } = block
  const v = vc ?? 0
  const values = rows.map((r) => toNum(r[v]) ?? 0)
  // Domain spans the actual min/max (anchored at 0) so negative values map
  // below the zero line instead of being pushed off the top.
  const maxVal = Math.max(0, ...values)
  const minVal = Math.min(0, ...values)
  const n = rows.length
  const width = Math.max(340, n * 80 + 60)
  const height = 240
  const padL = 40
  const padR = 20
  const padT = 18
  const padB = 40
  const plotW = width - padL - padR
  const plotH = height - padT - padB
  const domain = maxVal - minVal || 1
  const xs = rows.map((_, i) => padL + (i * plotW) / Math.max(1, n - 1))
  const ys = values.map((val) => padT + plotH - ((val - minVal) / domain) * plotH)
  const pts = rows.map((_, i) => `${xs[i]},${ys[i]}`).join(' ')
  const step = Math.max(1, Math.ceil(n / 8))

  return (
    <div className="chat-viz-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={block.title || 'line chart'}>
        <polyline
          points={pts}
          fill="none"
          stroke={COLORS[0]}
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {values.map((v, i) => (
          <g key={i}>
            <circle cx={xs[i]} cy={ys[i]} r={4} fill={COLORS[0]} stroke="#fff" strokeWidth={1.5} />
            <text x={xs[i]} y={ys[i] - 8} textAnchor="middle" className="chat-viz-bar-val">
              {formatValue(v, format)}
            </text>
          </g>
        ))}
        {rows.map((r, i) =>
          i % step === 0 || i === n - 1 ? (
            <text key={i} x={xs[i]} y={height - padB + 16} textAnchor="middle" className="chat-viz-bar-label">
              {String(r[0])}
            </text>
          ) : null,
        )}
      </svg>
    </div>
  )
}

function PieChart({ block }: { block: DataVizBlock }) {
  const { rows, columns, value_column: vc, format } = block
  const v = vc ?? 0
  const values = rows.map((r) => toNum(r[v]) ?? 0)
  const total = values.reduce((a, b) => a + b, 0)
  const cx = 110
  const cy = 110
  const r = 88
  let angle = -Math.PI / 2

  const arc = (start: number, end: number): string => {
    const x1 = cx + r * Math.cos(start)
    const y1 = cy + r * Math.sin(start)
    const x2 = cx + r * Math.cos(end)
    const y2 = cy + r * Math.sin(end)
    const large = end - start > Math.PI ? 1 : 0
    return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`
  }

  return (
    <div className="chat-viz-pie">
      <div className="chat-viz-chart">
        {total <= 0 ? (
          <div className="chat-viz-empty">No numeric data to plot.</div>
        ) : (
          <svg viewBox="0 0 220 220" role="img" aria-label={block.title || 'pie chart'}>
            {values.map((v, i) => {
              const slice = (v / total) * Math.PI * 2
              const d = arc(angle, angle + slice)
              const pct = total ? Math.round((v / total) * 1000) / 10 : 0
              const mid = angle + slice / 2
              const lx = cx + (r * 0.62) * Math.cos(mid)
              const ly = cy + (r * 0.62) * Math.sin(mid)
              angle += slice
              return (
                <g key={i}>
                  <path d={d} fill={COLORS[i % COLORS.length]} stroke="#fff" strokeWidth={1} />
                  {pct >= 4 && (
                    <text x={lx} y={ly} textAnchor="middle" dominantBaseline="central" className="chat-viz-pie-pct">
                      {pct}%
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        )}
      </div>
      <ol className="chat-viz-legend">
        {rows.map((row, i) => (
          <li key={i}>
            <span className="chat-viz-swatch" style={{ background: COLORS[i % COLORS.length] }} />
            <span className="chat-viz-legend-label">{String(row[0])}</span>
            <span className="chat-viz-legend-val">
              {formatValue(toNum(row[v]) ?? 0, format)} · {total ? Math.round((toNum(row[v])! / total) * 1000) / 10 : 0}%
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function PictogramChart({ block }: { block: DataVizBlock }) {
  const { rows, columns, value_column: vc, format } = block
  const v = vc ?? 0
  const values = rows.map((r) => toNum(r[v]) ?? 0)
  const maxVal = Math.max(1, ...values)
  const scale = maxVal > 40 ? Math.ceil(maxVal / 40) : 1

  return (
    <div className="chat-viz-picto">
      {rows.map((row, i) => {
        const val = toNum(row[v]) ?? 0
        const icons = Math.round(val / scale)
        return (
          <div className="chat-viz-picto-row" key={i}>
            <span className="chat-viz-picto-label">{String(row[0])}</span>
            <span className="chat-viz-picto-icons">
              {Array.from({ length: Math.max(0, icons) }).map((_, j) => (
                <span key={j} className="chat-viz-picto-icon" style={{ background: COLORS[i % COLORS.length] }} />
              ))}
            </span>
            <span className="chat-viz-picto-val">
              {formatValue(val, format)}
              {scale > 1 ? ` (${scale} per icon)` : ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function RenderView({ block, view }: { block: DataVizBlock; view: NonNullable<DataVizBlock['view']> }) {
  switch (view) {
    case 'bar':
      return <BarChart block={block} />
    case 'line':
      return <LineChart block={block} />
    case 'pie':
      return <PieChart block={block} />
    case 'picto':
      return <PictogramChart block={block} />
    default:
      return (
        <div className="chat-viz-table-wrap">
          <table className="chat-viz-table">
            <thead>
              <tr>
                {block.columns.map((c, ci) => (
                  <th key={ci}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>
                      {j === block.value_column && (isMissing(cell) ? '—' : toNum(cell) != null ? formatValue(toNum(cell)!, block.format) : String(cell))}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
  }
}

export default function DataViz({ block }: { block: DataVizBlock }) {
  const [view, setView] = useState<NonNullable<DataVizBlock['view']>>(block.kind ?? 'table')
  // Re-validate the selected view whenever the underlying block changes so a
  // stale selection (e.g. 'bar' left over from prior data) doesn't persist.
  useEffect(() => {
    setView(block.kind ?? 'table')
  }, [block])
  const vc = block.value_column
  // A chart is valid as long as at least one numeric value exists; missing
  // cells are skipped/zeroed per-series rather than forcing a table-only view.
  const numbers = useMemo(
    () => vc != null && block.rows.some((r) => toNum(r[vc]) != null),
    [block, vc],
  )
  const pictoOk = useMemo(() => {
    if (vc == null) return false
    const present = block.rows.map((r) => toNum(r[vc])).filter((x) => x != null) as number[]
    // Tolerate missing cells, but every present value must be a non-negative
    // integer for the pictogram to make sense.
    return present.length > 0 && present.every((x) => Number.isInteger(x) && x >= 0)
  }, [block, vc])
  // When the user explicitly asked for one view (block.view set by the backend),
  // render ONLY that view; otherwise keep the interactive view toggles.
  const locked = block.view
  const effective: NonNullable<DataVizBlock['view']> = useMemo(() => {
    const target = locked ?? view
    if (target === 'picto' && !pictoOk) return 'table'
    if ((target === 'bar' || target === 'line' || target === 'pie') && !numbers) return 'table'
    return target
  }, [locked, view, numbers, pictoOk])

  return (
    <div className="chat-viz">
      {block.title ? <div className="chat-viz-title">{block.title}</div> : null}
      {!locked && (
        <div className="chat-viz-tools">
          <div className="chat-viz-toggle" role="group" aria-label="View">
            <button type="button" aria-pressed={view === 'table'} className={view === 'table' ? 'active' : ''} onClick={() => setView('table')}>
              Table
            </button>
            {numbers && (
              <>
                <button type="button" aria-pressed={view === 'bar'} className={view === 'bar' ? 'active' : ''} onClick={() => setView('bar')}>
                  Bar
                </button>
                <button type="button" aria-pressed={view === 'line'} className={view === 'line' ? 'active' : ''} onClick={() => setView('line')}>
                  Line
                </button>
                <button type="button" aria-pressed={view === 'pie'} className={view === 'pie' ? 'active' : ''} onClick={() => setView('pie')}>
                  Pie
                </button>
                {pictoOk && (
                  <button type="button" aria-pressed={view === 'picto'} className={view === 'picto' ? 'active' : ''} onClick={() => setView('picto')}>
                    Pictogram
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}

      <RenderView block={block} view={effective} />
    </div>
  )
}