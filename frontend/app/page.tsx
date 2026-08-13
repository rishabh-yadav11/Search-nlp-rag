'use client'

import { useState } from 'react'

type Result = {
  id: number
  title: string
  url: string
  published_date: string
  category: string
  score: number
}

type ResponseData = {
  query: string
  results?: Result[]
  sources?: Result[]
  answer?: string
  cached: boolean
  latency_ms: number
}

type Status =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; answer?: string; results: Result[] }
  | { kind: 'error'; message: string }

const API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  'http://localhost:8000'

export default function Page() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'search' | 'ask'>('search')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [meta, setMeta] = useState('')

  async function run() {
    const q = query.trim()
    if (!q) return
    setLoading(true)
    setMeta('')
    setStatus({ kind: 'loading' })

    const t0 = performance.now()
    try {
      const endpoint = mode === 'ask' ? '/ask' : '/search'
      const res = await fetch(`${API_BASE}${endpoint}?q=${encodeURIComponent(q)}&top_k=8`)
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(`${res.status}: ${detail}`)
      }
      const data: ResponseData = await res.json()
      setStatus({ kind: 'done', answer: data.answer, results: data.results || data.sources || [] })
      const cached = data.cached ? 'cached · ' : ''
      setMeta(`${cached}${data.latency_ms.toFixed(0)}ms server · ${(performance.now() - t0).toFixed(0)}ms round-trip`)
    } catch (err) {
      setStatus({ kind: 'error', message: err instanceof Error ? err.message : String(err) })
      setMeta('')
    } finally {
      setLoading(false)
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
            autoComplete="off"
          />
          <div className="mode-toggle">
            <button type="button" className={mode === 'search' ? 'active' : ''} onClick={() => setMode('search')}>
              SEARCH
            </button>
            <button type="button" className={mode === 'ask' ? 'active' : ''} onClick={() => setMode('ask')}>
              ASK
            </button>
          </div>
          <button className="go-btn" type="submit" disabled={loading}>
            Run
          </button>
        </form>
        <div className="meta-row">{meta}</div>
        <ResultBlock status={status} mode={mode} />
      </main>
    </div>
  )
}

function ResultBlock({ status, mode }: { status: Status; mode: 'search' | 'ask' }) {
  switch (status.kind) {
    case 'idle':
      return <div className="empty">Results will appear here.</div>
    case 'loading':
      return <div className="loading">Searching</div>
    case 'error':
      return <div className="error">Request failed: {status.message}</div>
    case 'done':
      return (
        <div>
          {mode === 'ask' && status.answer && <div className="answer-block">{status.answer}</div>}
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
        <div className="result" key={r.id || i}>
          <div className="idx">{i + 1}</div>
          <div className="body">
            <a href={r.url} target="_blank" rel="noopener noreferrer">
              {r.title}
            </a>
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