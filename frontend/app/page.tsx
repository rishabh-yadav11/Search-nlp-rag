'use client'

import { useEffect, useRef, useState } from 'react'

type Result = {
  id: number
  title: string
  url: string
  published_date: string
  category: string
  score: number
  author_names?: string[]
  industry_names?: string[]
  dealtype_names?: string[]
}

type ResponseData = {
  query: string
  results: Result[]
  answer?: string
  cached: boolean
  latency_ms: number
  note?: string
}

type Status =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; answer?: string; results: Result[]; note?: string }
  | { kind: 'error'; message: string }

type SortBy = 'relevance' | 'date_desc' | 'date_asc' | 'score'

const API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000')

const TIMEOUT_MS = 30_000

function friendlyMessage(status: number): string {
  switch (status) {
    case 400:
      return 'Invalid query. Please check your input.'
    case 429:
      return 'Too many requests. Please try again shortly.'
    case 503:
      return 'Service temporarily unavailable. Please try again later.'
    default:
      return 'Something went wrong. Please try again.'
  }
}

function sanitizeStrings(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const strings = value.filter((v): v is string => typeof v === 'string')
  return strings.length ? strings : undefined
}

function sanitizeResponse(raw: unknown): ResponseData {
  const data = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>
  const rawResults = Array.isArray(data.results) ? data.results : Array.isArray(data.sources) ? data.sources : []

  const results: Result[] = rawResults.map((item, i) => {
    const r = (typeof item === 'object' && item !== null ? item : {}) as Record<string, unknown>
    return {
      id: typeof r.id === 'number' ? r.id : i,
      title: typeof r.title === 'string' ? r.title : '',
      url: typeof r.url === 'string' ? r.url : '',
      published_date: typeof r.published_date === 'string' ? r.published_date : '',
      category: typeof r.category === 'string' ? r.category : '',
      score: typeof r.score === 'number' ? r.score : 0,
      author_names: sanitizeStrings(r.author_names),
      industry_names: sanitizeStrings(r.industry_names),
      dealtype_names: sanitizeStrings(r.dealtype_names),
    }
  })

  return {
    query: typeof data.query === 'string' ? data.query : '',
    results,
    answer: typeof data.answer === 'string' ? data.answer : undefined,
    cached: data.cached === true,
    latency_ms: typeof data.latency_ms === 'number' ? data.latency_ms : 0,
    note: typeof data.note === 'string' && data.note.length ? data.note : undefined,
  }
}

function isSafeUrl(url: string): boolean {
  if (!url) return false
  try {
    const base = typeof window !== 'undefined' ? window.location.href : undefined
    const parsed = new URL(url, base)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

export default function Page() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [meta, setMeta] = useState('')
  const [sortBy, setSortBy] = useState<SortBy>('relevance')
  const submittedRef = useRef<{ controller: AbortController } | null>(null)

  useEffect(() => {
    return () => submittedRef.current?.controller.abort()
  }, [])

  async function run() {
    const q = query.trim()
    if (!q || loading) return

    submittedRef.current?.controller.abort()
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort('timeout'), TIMEOUT_MS)
    const submitted = { controller }
    submittedRef.current = submitted

    setLoading(true)
    setMeta('')
    setStatus({ kind: 'loading' })

    const t0 = performance.now()
    try {
      const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}&top_k=8`, {
        signal: controller.signal,
      })

      if (!res.ok) {
        const detail = await res.text()
        console.error(`Search API ${res.status}: ${detail}`)
        setStatus({ kind: 'error', message: friendlyMessage(res.status) })
        return
      }

      const data = sanitizeResponse(await res.json())
      setStatus({ kind: 'done', answer: data.answer, results: data.results, note: data.note })
      const cached = data.cached ? 'cached · ' : ''
      setMeta(`${cached}${data.latency_ms.toFixed(0)}ms server · ${(performance.now() - t0).toFixed(0)}ms round-trip`)
    } catch (err) {
      if (!controller.signal.aborted) {
        console.error('Search API network error', err)
        setStatus({ kind: 'error', message: 'Something went wrong. Please try again.' })
      } else if (controller.signal.reason === 'timeout') {
        setStatus({ kind: 'error', message: 'Request timed out. Please try again.' })
      }
    } finally {
      clearTimeout(timer)
      if (submittedRef.current === submitted) {
        submittedRef.current = null
        setLoading(false)
      }
    }
  }

  return (
    <div>
      <header>
        <div className="brand">
          <span className="mark">VCC</span>
          <h1>VCCircle New Search</h1>
        </div>
      </header>

      <main>
        <form
          className="search-row"
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. fintech startups that raised Series A"
            aria-label="Search query"
            autoComplete="off"
          />
          <button className="go-btn" type="submit" disabled={loading} aria-label="Search">
            <svg
              className="search-icon"
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </button>
        </form>
        <div className="results-region" aria-live="polite" aria-busy={loading}>
          <div className="meta-row">{meta}</div>
          <ResultBlock status={status} sortBy={sortBy} onSort={setSortBy} />
        </div>
      </main>
    </div>
  )
}

function ResultBlock({
  status,
  sortBy,
  onSort,
}: {
  status: Status
  sortBy: SortBy
  onSort: (s: SortBy) => void
}) {
  switch (status.kind) {
    case 'idle':
      return <div className="empty">Results will appear here.</div>
    case 'loading':
      return <div className="loading">Searching</div>
    case 'error':
      return (
        <div className="error" role="alert">
          {status.message}
        </div>
      )
    case 'done':
      return (
        <div>
          {status.note && (
            <div className="weak-note" role="note">
              {status.note}
            </div>
          )}
          <div className="sort-row">
            <label htmlFor="sort-select">Sort by</label>
            <select
              id="sort-select"
              value={sortBy}
              onChange={(e) => onSort(e.target.value as SortBy)}
            >
              <option value="relevance">Relevance</option>
              <option value="date_desc">Newest first</option>
              <option value="date_asc">Oldest first</option>
              <option value="score">Score</option>
            </select>
          </div>
          {renderResults(sortResults(status.results, sortBy))}
        </div>
      )
  }
}

function formatDate(s: string): string {
  if (!s) return 'n/a'
  const d = new Date(s)
  if (isNaN(d.getTime())) return s
  return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })
}

function relevance(score: number): { label: string; cls: string } {
  if (score >= 0.8) return { label: 'High', cls: 'high' }
  if (score >= 0.5) return { label: 'Medium', cls: 'medium' }
  return { label: 'Low', cls: 'low' }
}

function sortResults(items: Result[], sortBy: SortBy): Result[] {
  const sorted = [...items]
  switch (sortBy) {
    case 'date_desc':
      return sorted.sort((a, b) => {
        const da = Date.parse(a.published_date)
        const db = Date.parse(b.published_date)
        if (!isNaN(da) && !isNaN(db)) return db - da
        if (!isNaN(db)) return 1
        if (!isNaN(da)) return -1
        return 0
      })
    case 'date_asc':
      return sorted.sort((a, b) => {
        const da = Date.parse(a.published_date)
        const db = Date.parse(b.published_date)
        if (!isNaN(da) && !isNaN(db)) return da - db
        if (!isNaN(db)) return -1
        if (!isNaN(da)) return 1
        return 0
      })
    case 'score':
      return sorted.sort((a, b) => b.score - a.score)
    case 'relevance':
    default:
      return sorted
  }
}

function renderResults(items: Result[]) {
  if (!items.length) return <div className="empty">No matches found.</div>
  return (
    <div>
      <div className="results-heading">Results</div>
      {items.map((r, i) => {
        const rel = relevance(r.score)
        return (
          <div className="result" key={r.id}>
            <div className="idx">{i + 1}</div>
            <div className="body">
              {isSafeUrl(r.url) ? (
                <a href={r.url} target="_blank" rel="noopener noreferrer">
                  {r.title || 'Untitled'}
                </a>
              ) : (
                <span className="plain-title">{r.title || 'Untitled'}</span>
              )}
              <div className="info">
                <span className={`badge ${rel.cls}`}>{rel.label}</span>
                <span className="score">{r.score.toFixed(3)}</span>
                <span>{formatDate(r.published_date)}</span>
                {r.category ? <span>{r.category}</span> : null}
              </div>
              <div className="facet-line">
                {r.author_names?.length ? <span className="facet">✎ {r.author_names.join(', ')}</span> : null}
                {r.industry_names?.length ? <span className="facet">◆ {r.industry_names.join(', ')}</span> : null}
                {r.dealtype_names?.length ? <span className="facet">● {r.dealtype_names.join(', ')}</span> : null}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
