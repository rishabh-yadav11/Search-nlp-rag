'use client'

import { useMemo, useState } from 'react'

export type DataVizBlock = {
  title?: string
  columns: string[]
  rows: (string | number)[][]
  value_column: number
  format?: string
}

type ContentPart = { type: 'md'; md: string } | { type: 'viz'; block: DataVizBlock }

const FENCE_SRC = '```dataviz\\s*\\n([\\s\\S]*?)\\n```'

function toNum(v: unknown): number | null {
  if (typeof v === 'boolean') return null
  if (typeof v === 'number') return v
  if (typeof v === 'string') {
    const n = parseFloat(v.replace(/,/g, ''))
    return Number.isFinite(n) ? n : null
  }
  return null
}

function firstNumericColumn(rows: (string | number)[][]): number | null {
  if (!rows.length || !rows[0].length) return null
  for (let j = 0; j < rows[0].length; j++) {
    if (rows.every((r) => toNum(r[j]) != null)) return j
  }
  return null
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
    if (vc == null || rowArr.some((r) => toNum(r[vc]) == null)) return null
    return {
      title: typeof (d as { title?: unknown }).title === 'string' ? (d as { title: string }).title : undefined,
      columns: columns as string[],
      rows: rowArr,
      value_column: vc as number,
      format: typeof (d as { format?: unknown }).format === 'string' ? (d as { format: string }).format : undefined,
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
      return v.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
}

function trim(v: number): string {
  const s = Math.round(v * 100) / 100
  return Number.isInteger(s) ? String(s) : String(s)
}

const COLORS = ['#1a5fb4', '#26a269', '#e66100', '#c01c28', '#613583', '#2a7bde', '#d67600', '#8f5902', '#4e9a06', '#75507b']

function BarChart({ block }: { block: DataVizBlock }) {
  const { rows, columns, value_column: vc, format } = block
  const values = rows.map((r) => toNum(r[vc]) ?? 0)
  const maxVal = Math.max(1, ...values)
  const n = rows.length
  const slot = 64
  const width = Math.max(320, n * slot + 60)
  const height = 240
  const plotTop = 18
  const plotBottom = height - 66
  const plotH = plotBottom - plotTop
  const barW = Math.min(44, slot - 16)
  const rotate = n > 4

  return (
    <div className="chat-viz-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={block.title || 'bar chart'}>
        {values.map((v, i) => {
          const bh = Math.max(2, (v / maxVal) * plotH)
          const x = 30 + i * slot
          const y = plotBottom - bh
          const label = String(rows[i][0] ?? '').length > 14 ? `${String(rows[i][0] ?? '').slice(0, 13)}…` : String(rows[i][0] ?? '')
          return (
            <g key={i}>
              <rect x={x} y={y} width={barW} height={bh} rx={3} fill={COLORS[i % COLORS.length]} />
              <text x={x + barW / 2} y={y - 5} textAnchor="middle" className="chat-viz-bar-val">
                {formatValue(v, format)}
              </text>
              <text
                x={x + barW / 2}
                y={height - 14}
                textAnchor="end"
                transform={`rotate(${rotate ? -24 : 0} ${x + barW / 2} ${height - 14})`}
                className="chat-viz-bar-label"
              >
                <title>{String(rows[i][0] ?? '')}</title>
                {label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function PieChart({ block }: { block: DataVizBlock }) {
  const { rows, columns, value_column: vc, format } = block
  const values = rows.map((r) => toNum(r[vc]) ?? 0)
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
              {formatValue(toNum(row[vc]) ?? 0, format)} · {total ? Math.round((toNum(row[vc])! / total) * 1000) / 10 : 0}%
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

export default function DataViz({ block }: { block: DataVizBlock }) {
  const [view, setView] = useState<'table' | 'bar' | 'pie'>('table')
  const [raw, setRaw] = useState(false)
  const numbers = useMemo(
    () => (block.rows.every((r) => toNum(r[block.value_column]) != null) ? true : false),
    [block],
  )

  return (
    <div className="chat-viz">
      {block.title ? <div className="chat-viz-title">{block.title}</div> : null}
      <div className="chat-viz-tools">
        <div className="chat-viz-toggle" role="group" aria-label="View">
          <button type="button" className={view === 'table' ? 'active' : ''} onClick={() => setView('table')}>
            Table
          </button>
          {numbers && (
            <>
              <button type="button" className={view === 'bar' ? 'active' : ''} onClick={() => setView('bar')}>
                Bar
              </button>
              <button type="button" className={view === 'pie' ? 'active' : ''} onClick={() => setView('pie')}>
                Pie
              </button>
            </>
          )}
        </div>
        <button type="button" className="chat-viz-raw-btn" onClick={() => setRaw((r) => !r)}>
          {raw ? 'Hide raw data' : 'Raw data'}
        </button>
      </div>

      {view === 'table' && (
        <div className="chat-viz-table-wrap">
          <table className="chat-viz-table">
            <thead>
              <tr>
                {block.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>
                      {j === block.value_column && toNum(cell) != null
                        ? formatValue(toNum(cell)!, block.format)
                        : String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {view === 'bar' && numbers && <BarChart block={block} />}
      {view === 'pie' && numbers && <PieChart block={block} />}

      {raw && <pre className="chat-viz-raw">{JSON.stringify(block, null, 2)}</pre>}
    </div>
  )
}
