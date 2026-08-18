'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import {
  AuthUser,
  authHeaders,
  clearMeCache,
  getMe,
  getToken,
  redirectToLogin,
} from './lib/auth'

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
  summary?: string
}

type ResponseData = {
  query: string
  results: Result[]
  cached: boolean
  latency_ms: number
  note?: string
}

type Status =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; query: string; results: Result[]; note?: string; hint?: string }
  | { kind: 'error'; message: string }

type SortBy = 'relevance' | 'date_desc' | 'date_asc' | 'score'

type Filters = {
  industry: string
  dealtype: string
  from_date: string
  to_date: string
}

const EMPTY_FILTERS: Filters = {
  industry: '',
  dealtype: '',
  from_date: '',
  to_date: '',
}

const SUGGESTIONS = [
  'top 10 fintech deals 2025',
  'Ola Electric IPO',
  'top venture debt providers 2024',
]

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
  const rawResults = Array.isArray(data.results) ? data.results : []

  const results: Result[] = rawResults.map((item, i) => {
    const r = (typeof item === 'object' && item !== null ? item : {}) as Record<string, unknown>
    const summary = typeof r.summary === 'string' ? r.summary.trim() : ''
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
      summary: summary || undefined,
    }
  })

  return {
    query: typeof data.query === 'string' ? data.query : '',
    results,
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

function trackClick(query: string, position: number) {
  if (typeof navigator === 'undefined' || !query) return
  try {
    const payload = JSON.stringify({ query, position })
    const sent = navigator.sendBeacon(`${API_BASE}/analytics/click`, new Blob([payload], { type: 'application/json' }))
    if (sent) return
    fetch(`${API_BASE}/analytics/click`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
      keepalive: true,
    }).catch(() => {})
  } catch {
    /* analytics is best-effort; never block navigation */
  }
}

export default function Page() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [meta, setMeta] = useState('')
  const [sortBy, setSortBy] = useState<SortBy>('relevance')
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS)
  const [showFilters, setShowFilters] = useState(false)
  const [hideLow, setHideLow] = useState(false)
  const [facetOptions, setFacetOptions] = useState<{ industry: string[]; dealtype: string[] }>({
    industry: [],
    dealtype: [],
  })
  const submittedRef = useRef<{ controller: AbortController } | null>(null)
  const [me, setMe] = useState<AuthUser | null | undefined>(undefined)

  useEffect(() => {
    if (!getToken()) {
      setMe(null)
      return
    }
    let cancelled = false
    getMe()
      .then((u) => {
        if (!cancelled) setMe(u)
      })
      .catch(() => {
        if (!cancelled) setMe(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  function logout() {
    fetch(`${API_BASE}/api/auth/logout`, { method: 'POST', headers: authHeaders() }).catch(() => {})
    clearMeCache()
    redirectToLogin()
  }

  useEffect(() => {
    return () => submittedRef.current?.controller.abort()
  }, [])

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/facets`, { signal: AbortSignal.timeout(10_000) })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data) return
        setFacetOptions({
          industry: Array.isArray(data.industry) ? data.industry : [],
          dealtype: Array.isArray(data.dealtype) ? data.dealtype : [],
        })
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const hasFilters = Object.values(filters).some((v) => v.trim() !== '')

  function updateFilter(key: keyof Filters, value: string) {
    setFilters((f) => ({ ...f, [key]: value }))
  }

  async function run(qOverride?: string) {
    const q = (qOverride ?? query).trim()
    if (loading) return
    if (!q) {
      setMeta('')
      setStatus({ kind: 'done', query: '', results: [], hint: 'Please enter a query to search.' })
      return
    }

    submittedRef.current?.controller.abort()
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort('timeout'), TIMEOUT_MS)
    const submitted = { controller }
    submittedRef.current = submitted

    setLoading(true)
    setMeta('')
    setStatus({ kind: 'loading' })

    const params = new URLSearchParams({ q, top_k: '8' })
    for (const [key, value] of Object.entries(filters)) {
      if (value && value.trim()) params.set(key, value.trim())
    }

    const t0 = performance.now()
    try {
      const res = await fetch(`${API_BASE}/search?${params.toString()}`, {
        signal: controller.signal,
      })

      if (!res.ok) {
        const detail = await res.text()
        console.error(`Search API ${res.status}: ${detail}`)
        setStatus({ kind: 'error', message: friendlyMessage(res.status) })
        return
      }

      const data = sanitizeResponse(await res.json())
      setStatus({ kind: 'done', query: data.query || q, results: data.results, note: data.note })
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

  function handleSuggestion(s: string) {
    setQuery(s)
    run(s)
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="logo" src="/vccircle-wordmark.svg" alt="VCCircle" width={154} height={40} />
          </div>
          <nav className="topbar-nav" aria-label="Primary">
            <Link href="/" className="topbar-nav-link active">
              Search
            </Link>
            <Link href="/chat" className="topbar-nav-link">
              Chat
            </Link>
            {me?.role === 'admin' ? (
              <a href="/analytics/dashboard" className="topbar-nav-link">
                Analytics
              </a>
            ) : null}
          </nav>
          <div className="topbar-right">
            <Link href="/chat" className="topbar-cta" aria-label="Open chat assistant">
              ASK VCCircle
            </Link>
            {me === undefined ? null : me ? (
              <span className="topbar-user">
                <span className="topbar-user-email" title={me.email}>
                  {me.name || me.email}
                </span>
                <button type="button" className="topbar-logout" onClick={logout}>
                  Log out
                </button>
              </span>
            ) : (
              <Link href="/login" className="topbar-signin">
                Sign in
              </Link>
            )}
          </div>
        </div>
      </header>

      <main>
        <form
          id="search"
          className="search-hero search-row"
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
          <button
            type="button"
            className={`filter-toggle ${showFilters ? 'active' : ''}`}
            onClick={() => setShowFilters((v) => !v)}
            aria-expanded={showFilters}
            aria-controls="filters-panel"
            aria-label={showFilters ? 'Hide filters' : 'Show filters'}
          >
            <svg
              className="filter-icon"
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
              <path d="M4 6h16" />
              <path d="M7 12h10" />
              <path d="M10 18h4" />
            </svg>
          </button>
        </form>

        <div className="suggestions" role="group" aria-label="Search suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="chip" onClick={() => handleSuggestion(s)} aria-label={`Search for ${s}`}>
              {s}
            </button>
          ))}
        </div>

        {showFilters && (
          <section id="filters-panel" className="filters" aria-label="Search filters">
            <div className="filters-grid">
                <label className="filter-field">
                  <span className="filter-label">industry</span>
                  <input
                    type="text"
                    list="industry-options"
                    value={filters.industry}
                    onChange={(e) => updateFilter('industry', e.target.value)}
                    placeholder="e.g. fintech"
                    aria-label="Industry filter"
                  />
                </label>
                <datalist id="industry-options">
                  {facetOptions.industry.map((v) => (
                    <option key={v} value={v} />
                  ))}
                </datalist>
                <label className="filter-field">
                  <span className="filter-label">dealtype</span>
                  <input
                    type="text"
                    list="dealtype-options"
                    value={filters.dealtype}
                    onChange={(e) => updateFilter('dealtype', e.target.value)}
                    placeholder="e.g. venture debt"
                    aria-label="Dealtype filter"
                  />
                </label>
                <datalist id="dealtype-options">
                  {facetOptions.dealtype.map((v) => (
                    <option key={v} value={v} />
                  ))}
                </datalist>
                <label className="filter-field">
                  <span className="filter-label">from_date</span>
                  <input
                    type="date"
                    value={filters.from_date}
                    onChange={(e) => updateFilter('from_date', e.target.value)}
                    aria-label="From date filter"
                  />
                </label>
                <label className="filter-field">
                  <span className="filter-label">to_date</span>
                  <input
                    type="date"
                    value={filters.to_date}
                    onChange={(e) => updateFilter('to_date', e.target.value)}
                    aria-label="To date filter"
                  />
                </label>
                {hasFilters && (
                  <button type="button" className="clear-filters" onClick={() => setFilters(EMPTY_FILTERS)}>
                    Clear filters
                  </button>
                )}
              </div>
            </section>
          )}

        <div className="results-region" aria-live="polite" aria-busy={loading}>
          <div className="meta-row">{meta}</div>
          <ResultBlock status={status} sortBy={sortBy} onSort={setSortBy} hideLow={hideLow} onHideLow={setHideLow} />
        </div>
      </main>

      <footer className="site-footer">
        <div className="footer-inner">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="footer-logo" src="/vccircle-wordmark.svg" alt="VCCircle" width={120} height={32} />
          <span className="footer-note">
            ASK VCCircle &middot; AI-assisted search over VCCircle&rsquo;s archives
          </span>
          <span className="footer-copy">&copy; 2026 VCCircle</span>
        </div>
      </footer>
    </div>
  )
}

function ResultBlock({
  status,
  sortBy,
  onSort,
  hideLow,
  onHideLow,
}: {
  status: Status
  sortBy: SortBy
  onSort: (s: SortBy) => void
  hideLow: boolean
  onHideLow: (v: boolean) => void
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
      if (status.hint) {
        return (
          <div className="hint" role="status">
            {status.hint}
          </div>
        )
      }
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
            <label className="hide-low" htmlFor="hide-low-input">
              <input
                id="hide-low-input"
                type="checkbox"
                checked={hideLow}
                onChange={(e) => onHideLow(e.target.checked)}
              />
              Hide low relevance
            </label>
          </div>
          {renderResults(filterByRelevance(sortResults(status.results, sortBy), hideLow), status.query)}
        </div>
      )
  }
}

function formatDate(s: string): string {
  if (!s) return 'n/a'
  const d = new Date(s)
  if (isNaN(d.getTime())) return s
  const now = Date.now()
  const diff = now - d.getTime()
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (diff < minute) return 'just now'
  if (diff < hour) return `${Math.floor(diff / minute)} minute${Math.floor(diff / minute) > 1 ? 's' : ''} ago`
  if (diff < day) return `${Math.floor(diff / hour)} hour${Math.floor(diff / hour) > 1 ? 's' : ''} ago`
  if (diff < 2 * day) return 'yesterday'
  if (diff < 7 * day) return `${Math.floor(diff / day)} days ago`
  if (diff < 30 * day) return `${Math.floor(diff / (7 * day))} week${Math.floor(diff / (7 * day)) > 1 ? 's' : ''} ago`
  if (diff < 365 * day) return `${Math.floor(diff / (30 * day))} month${Math.floor(diff / (30 * day)) > 1 ? 's' : ''} ago`
  return `${Math.floor(diff / (365 * day))} year${Math.floor(diff / (365 * day)) > 1 ? 's' : ''} ago`
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

function filterByRelevance(items: Result[], hideLow: boolean): Result[] {
  if (!hideLow) return items
  return items.filter((r) => r.score >= 0.5)
}

const FACET_PLACEHOLDERS = new Set(['others', 'general', 'n/a', 'na', 'unknown', 'none', 'miscellaneous'])

function cleanFacets(values: string[] | undefined): string {
  if (!values?.length) return ''
  return values.filter((v) => !FACET_PLACEHOLDERS.has(v.trim().toLowerCase())).join(', ')
}

function renderResults(items: Result[], query: string) {
  if (!items.length) return <div className="empty">No matches found.</div>
  return (
    <div>
      <div className="results-heading">
        {items.length} results for &ldquo;<span className="query">{query}</span>&rdquo;
      </div>
      {items.map((r, i) => {
        const rel = relevance(r.score)
        const authors = cleanFacets(r.author_names)
        const industries = cleanFacets(r.industry_names)
        const dealtypes = cleanFacets(r.dealtype_names)
        return (
          <div className="result" key={r.id}>
            <div className="idx">{i + 1}</div>
            <div className="body">
              {isSafeUrl(r.url) ? (
                <a
                  href={r.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => trackClick(query, i + 1)}
                >
                  {r.title || 'Untitled'}
                </a>
              ) : (
                <span className="plain-title">{r.title || 'Untitled'}</span>
              )}
              {r.summary ? <p className="excerpt">{r.summary}</p> : null}
              <div className="info">
                <span className={`badge ${rel.cls}`}>{rel.label}</span>
                <span className="score">{r.score.toFixed(3)}</span>
                <span>{formatDate(r.published_date)}</span>
                {r.category && !FACET_PLACEHOLDERS.has(r.category.trim().toLowerCase()) ? <span>{r.category}</span> : null}
              </div>
              <div className="facet-line">
                {authors ? <span className="facet">✎ {authors}</span> : null}
                {industries ? <span className="facet">◆ {industries}</span> : null}
                {dealtypes ? <span className="facet">● {dealtypes}</span> : null}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
