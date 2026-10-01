'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createPaginatedRowModel,
  flexRender,
  rowPaginationFeature,
  tableFeatures,
  type ColumnDef,
  useTable,
} from '@tanstack/react-table'
import {
  Bar,
  BarChart as ReBarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart as ReLineChart,
  Pie,
  PieChart as RePieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// The accept/reject grammar lives in datavizContract.ts, which the backend
// executes through node so the two validators cannot drift apart. Only
// rendering lives here.
import { isMissing, toNum } from './datavizContract'
import type { DataVizBlock } from './datavizContract'

// Re-exported so the chat page and its tests keep importing them from here.
export type { DataVizBlock } from './datavizContract'
export { parseDataViz, splitContent } from './datavizContract'

function labelColumnIndex(columns: string[], valueColumn: number | null): number {
  const v = valueColumn ?? 0
  const j = columns.findIndex((_, k) => k !== v)
  return j >= 0 ? j : 0
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

function chartData(block: DataVizBlock): { label: string; value: number | null }[] {
  const { rows, value_column: vc, columns } = block
  const v = vc ?? 0
  const lc = labelColumnIndex(columns, vc)
  const label = (r: (string | number)[]) => (isMissing(r[lc]) ? '' : String(r[lc]))
  const num = (r: (string | number)[]) => (isMissing(r[v]) ? null : toNum(r[v]))
  return rows.map((r) => ({ label: label(r), value: num(r) }))
}

const VALUE_KEY = 'value'
const LABEL_KEY = 'label'

const COLORS = [
  '#0072b2', '#e69f00', '#009e73', '#d55e00', '#cc79a7', '#56b4e9',
  '#007777', '#e07b00', '#332288', '#44aa99', '#882255', '#88ccee',
  '#6699cc', '#aa4499', '#117733', '#ddaa33', '#55aa77', '#bb5566',
  '#336699', '#cc6600', '#2299aa', '#ee7733', '#6b4a99', '#446688',
]

function BarChart({ block }: { block: DataVizBlock }) {
  const data = chartData(block)
  const format = block.format
  return (
    <div className="chat-viz-chart">
      <ResponsiveContainer width="100%" height={260} minWidth={300}>
        <ReBarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ececec" vertical={false} />
          <XAxis
            dataKey={LABEL_KEY}
            tick={{ fontSize: 10.5, fill: '#444' }}
            interval={0}
            angle={data.length > 6 ? -20 : 0}
            textAnchor={data.length > 6 ? 'end' : 'middle'}
            height={data.length > 6 ? 52 : 30}
          />
          <YAxis tick={{ fontSize: 10, fill: '#888' }} width={48} />
          <Tooltip
            formatter={(value) => [formatValue(Number(value), format), '']}
            labelStyle={{ fontSize: 12, color: '#333' }}
            contentStyle={{ fontSize: 12 }}
          />
          <Bar dataKey={VALUE_KEY} radius={[3, 3, 0, 0]} maxBarSize={44}>
            {data.map((d, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </ReBarChart>
      </ResponsiveContainer>
    </div>
  )
}

function LineChart({ block }: { block: DataVizBlock }) {
  const data = chartData(block)
  const format = block.format
  return (
    <div className="chat-viz-chart">
      <ResponsiveContainer width="100%" height={260} minWidth={300}>
        <ReLineChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ececec" vertical={false} />
          <XAxis dataKey={LABEL_KEY} tick={{ fontSize: 10.5, fill: '#444' }} interval="preserveStartEnd" height={30} />
          <YAxis tick={{ fontSize: 10, fill: '#888' }} width={48} />
          <Tooltip
            formatter={(value) => [formatValue(Number(value), format), '']}
            labelStyle={{ fontSize: 12, color: '#333' }}
            contentStyle={{ fontSize: 12 }}
          />
          <Line
            type="monotone"
            dataKey={VALUE_KEY}
            stroke={COLORS[0]}
            strokeWidth={2.5}
            dot={{ r: 4, fill: COLORS[0], stroke: '#fff', strokeWidth: 1.5 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </ReLineChart>
      </ResponsiveContainer>
    </div>
  )
}

function PieChart({ block }: { block: DataVizBlock }) {
  const data = chartData(block)
    .map((d) => ({ ...d, value: d.value == null ? 0 : Math.max(0, d.value) }))
    .filter((d) => d.value > 0)
  const format = block.format
  const total = data.reduce((a, d) => a + d.value, 0)
  return (
    <div className="chat-viz-pie">
      <div className="chat-viz-chart">
        {total <= 0 ? (
          <div className="chat-viz-empty">No numeric data to plot.</div>
        ) : (
          <ResponsiveContainer width="100%" height={260} minWidth={300}>
            <RePieChart>
              <Pie
                data={data}
                dataKey={VALUE_KEY}
                nameKey={LABEL_KEY}
                cx="50%"
                cy="50%"
                outerRadius={88}
                innerRadius={60}
                label={({ value }) => {
                  const pct = total ? Math.round((value / total) * 1000) / 10 : 0
                  return pct >= PIE_LABEL_MIN_PCT ? `${pct}%` : ''
                }}
                labelLine={false}
                fill="#8884d8"
              >
                {data.map((d, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => [formatValue(Number(value), format), '']}
                labelStyle={{ fontSize: 12, color: '#333' }}
                contentStyle={{ fontSize: 12 }}
              />
              <Legend
                iconType="circle"
                iconSize={10}
                layout="vertical"
                verticalAlign="middle"
                align="right"
                wrapperStyle={{ fontSize: 12, paddingLeft: 12 }}
              />
            </RePieChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}

const TABLE_PAGE_SIZE = 25
const TABLE_PAGINATE_AT = 50
const PIE_LABEL_MIN_PCT = 5

const vizFeatures = tableFeatures({
  rowPaginationFeature,
  paginatedRowModel: createPaginatedRowModel(),
})

function TableView({ block }: { block: DataVizBlock }) {
  const columns = useMemo<ColumnDef<typeof vizFeatures, (string | number)[], unknown>[]>(() =>
    block.columns.map((col, ci) => ({
      accessorKey: String(ci),
      header: col,
      cell: (info) => {
        const cell = (info.row.original as (string | number)[])[ci]
        if (ci === block.value_column) {
          if (isMissing(cell)) return '—'
          const n = toNum(cell)
          return n != null ? formatValue(n, block.format) : String(cell)
        }
        return String(cell)
      },
    })),
    [block],
  )

  const table = useTable({
    features: vizFeatures,
    data: block.rows,
    columns,
    initialState: { pagination: { pageIndex: 0, pageSize: TABLE_PAGE_SIZE } },
  })

  const hasControls = block.rows.length > TABLE_PAGINATE_AT

  return (
    <div className="chat-viz-table-wrap">
      <table className="chat-viz-table">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => (
                <th key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id}>
              {row.getAllCells().map((cell) => (
                <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {hasControls && (
        <div className="chat-viz-table-nav">
          <button
            type="button"
            className="chat-viz-table-nav-btn"
            disabled={!table.getCanPreviousPage()}
            onClick={() => table.previousPage()}
            aria-label="Previous page"
          >
            ‹ Prev
          </button>
          <span className="chat-viz-table-nav-page" aria-live="polite">
            Page {table.state.pagination.pageIndex + 1} of {table.getPageCount()}
          </span>
          <button
            type="button"
            className="chat-viz-table-nav-btn"
            disabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
            aria-label="Next page"
          >
            Next ›
          </button>
        </div>
      )}
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
    default:
      return <TableView block={block} />
  }
}

export default function DataViz({ block }: { block: DataVizBlock }) {
  const [view, setView] = useState<NonNullable<DataVizBlock['view']>>(block.kind ?? 'table')
  const prevSig = useRef<string>('')
  const sig = `${block.title ?? ''}|${block.rows.length}|${block.value_column ?? ''}|${block.kind ?? ''}`
  useEffect(() => {
    if (prevSig.current === sig) return
    prevSig.current = sig
    setView(block.kind ?? 'table')
  }, [sig, block.kind])
  const vc = block.value_column
  const numbers = useMemo(
    () => vc != null && block.rows.some((r) => toNum(r[vc]) != null),
    [block, vc],
  )
  const locked = block.view
  const effective: NonNullable<DataVizBlock['view']> = useMemo(() => {
    const target = locked ?? view
    if ((target === 'bar' || target === 'line' || target === 'pie') && !numbers) return 'table'
    return target
  }, [locked, view, numbers])

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
              </>
            )}
          </div>
        </div>
      )}
      <RenderView block={block} view={effective} />
    </div>
  )
}