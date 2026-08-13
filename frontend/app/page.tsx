'use client'

import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type Mode = 'search' | 'ask'

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
}

type Status =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; mode: Mode; answer?: string; results: Result[] }
  | { kind: 'error'; mode: Mode; message: string }

const API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000')

const TIMEOUT_MS: Record<Mode, number> = { search: 30_000, ask: 60_000 }

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
  const [mode, setMode] = useState<Mode>('search')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [meta, setMeta] = useState('')
  const submittedRef = useRef<{ mode: Mode; controller: AbortController } | null>(null)

  useEffect(() => {
    return () => submittedRef.current?.controller.abort()
  }, [])

  async function run() {
    const q = query.trim()
    if (!q || loading) return

    submittedRef.current?.controller.abort()
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort('timeout'), TIMEOUT_MS[mode])
    const submitted = { mode, controller }
    submittedRef.current = submitted

    setLoading(true)
    setMeta('')
    setStatus({ kind: 'loading' })

    const t0 = performance.now()
    try {
      const endpoint = mode === 'ask' ? '/ask' : '/search'
      const res = await fetch(`${API_BASE}${endpoint}?q=${encodeURIComponent(q)}&top_k=8`, {
        signal: controller.signal,
      })

      if (!res.ok) {
        const detail = await res.text()
        console.error(`Search API ${res.status}: ${detail}`)
        setStatus({ kind: 'error', mode, message: friendlyMessage(res.status) })
        return
      }

      const data = sanitizeResponse(await res.json())
      setStatus({ kind: 'done', mode, answer: data.answer, results: data.results })
      const cached = data.cached ? 'cached · ' : ''
      setMeta(`${cached}${data.latency_ms.toFixed(0)}ms server · ${(performance.now() - t0).toFixed(0)}ms round-trip`)
    } catch (err) {
      if (!controller.signal.aborted) {
        console.error('Search API network error', err)
        setStatus({ kind: 'error', mode, message: 'Something went wrong. Please try again.' })
      } else if (controller.signal.reason === 'timeout') {
        setStatus({ kind: 'error', mode, message: 'Request timed out. Please try again.' })
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
          <h1>Semantic Search — POC</h1>
        </div>
        <span className="tag">hybrid retrieval · optional cited answers</span>
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
          <div className="mode-toggle" role="group" aria-label="Search mode">
            <button
              type="button"
              aria-pressed={mode === 'search'}
              disabled={loading}
              className={mode === 'search' ? 'active' : ''}
              onClick={() => setMode('search')}
            >
              SEARCH
            </button>
            <button
              type="button"
              aria-pressed={mode === 'ask'}
              disabled={loading}
              className={mode === 'ask' ? 'active' : ''}
              onClick={() => setMode('ask')}
            >
              ASK
            </button>
          </div>
          <button className="go-btn" type="submit" disabled={loading}>
            Run
          </button>
        </form>
        <div className="results-region" aria-live="polite" aria-busy={loading}>
          <div className="meta-row">{meta}</div>
          <ResultBlock status={status} />
        </div>
      </main>
    </div>
  )
}

function ResultBlock({ status }: { status: Status }) {
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
          {status.mode === 'ask' && status.answer && (
            <div className="answer-block markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{status.answer}</ReactMarkdown>
            </div>
          )}
          {renderResults(status.results)}
        </div>
      )
  }
}

function renderResults(items: Result[]) {
  if (!items.length) return <div className="empty">No matches found.</div>
  return (
    <div>
      <div className="results-heading">Sources</div>
      {items.map((r, i) => (
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
              <span className="score">{r.score.toFixed(3)}</span>
              <span>{r.published_date || 'n/a'}</span>
              {r.category ? <span>{r.category}</span> : null}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
