'use client'

import { useEffect, useRef, useState } from 'react'
import { API_BASE, authRequestInit, getMe, redirectToLogin } from '../../lib/auth'
import type { AuthUser } from '../../lib/auth'
import TopBar from '../../components/TopBar'
import { formatClockTime, formatCost, formatEpochDateTime } from '../../lib/format'

interface Summary {
  searches_total: number
  searches_today: number
  zero_result_rate: number
  weak_result_rate: number
  filtered_rate: number
  cache_hit_rate: number
  avg_latency_ms: number
  clicks_total: number
  top_queries: [string, number][]
  click_positions: Record<string, number>
  click_top_queries: [string, number][]
}

// Row[0] is an opaque keyed digest, never the query text: a bare hash of a
// short query is reversible offline by anyone who can read this dashboard.

type ChatRow = [sessionId: string, messages: number, cost: number, updatedAt: number]
type ChatTokenRow = [sessionId: string, messages: number, tokens: number, updatedAt: number]

interface ChatStats {
  sessions: number
  users: number
  messages: number
  total_tokens: number
  total_cost: number
  avg_latency_ms: number
  sessions_today: number
  top_by_cost: ChatRow[]
  top_by_tokens: ChatTokenRow[]
}

function fmt(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? '0' : Number(n).toLocaleString()
}

function pct(v: number | null | undefined): string {
  return v == null ? '0%' : `${v}%`
}

// A failed feed is never rendered as zeros, which are the legitimate state of a quiet day.
type Feed<T> = { data: T } | { degraded: string }

/** A 200 carrying `error` is a failure, not data. */
async function readFeed<T>(res: Response, label: string): Promise<Feed<T>> {
  if (!res.ok) return { degraded: `${label} (HTTP ${res.status})` }
  let body: unknown
  try {
    body = await res.json()
  } catch {
    return { degraded: `${label} returned an unreadable response` }
  }
  if (body && typeof body === 'object' && 'error' in body) {
    const detail = body.error
    return {
      degraded: `${label}: ${typeof detail === 'string' ? detail : 'unavailable'}`,
    }
  }
  return { data: body as T }
}

function Card({ label, value, hint, warn }: { label: string; value: string; hint?: string; warn?: boolean }) {
  return (
    <div className="dash-card">
      <div className="dash-label">{label}</div>
      <div className={`dash-value ${warn ? 'warn' : ''}`}>{value}</div>
      {hint ? <div className="dash-hint">{hint}</div> : null}
    </div>
  )
}

function TopTable({ rows }: { rows: [string, number][] | undefined }) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No data yet.</div>
  const max = rows[0][1] || 1
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Query id</th>
          <th scope="col" className="num">Count</th>
          <th scope="col"></th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([q, n], i) => (
          <tr key={`topq-${i}-${q}`}>
            <td>{q}</td>
            <td className="num">{fmt(n)}</td>
            <td width="34%">
              <span className="dash-bar" style={{ width: `${Math.round((100 * n) / max)}%` }}></span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ChatTable({
  rows,
  cost,
}: {
  rows: (ChatRow | ChatTokenRow)[] | undefined
  cost: boolean
}) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No chat activity yet.</div>
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Session</th>
          <th scope="col" className="num">Msgs</th>
          <th scope="col" className="num">{cost ? 'Cost' : 'Tokens'}</th>
          <th scope="col">Updated</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([sessionId, msgs, value, ts]) => (
          <tr key={sessionId}>
            {/* Surrogate only: the conversation text behind this id must never reach the table. */}
            <td title={sessionId}>Session {sessionId.slice(0, 8)}</td>
            <td className="num">{fmt(msgs)}</td>
            <td className="num">{cost ? formatCost(value) : fmt(value)}</td>
            <td>{formatEpochDateTime(ts)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function AnalyticsDashboardPage() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [chat, setChat] = useState<ChatStats | null>(null)
  const [updated, setUpdated] = useState('loading…')
  const [error, setError] = useState('')
  const [forbidden, setForbidden] = useState(false)
  const [me, setMe] = useState<AuthUser | null | undefined>(undefined)

  const [degraded, setDegraded] = useState<{ summary: string | null; chat: string | null }>({
    summary: null,
    chat: null,
  })

  const inFlight = useRef(false)
  const controllerRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)
  const FETCH_TIMEOUT_MS = 15000
  const GETME_TIMEOUT_MS = 10000

  function withTimeout<T>(p: Promise<T>, ms: number): Promise<{ timedOut: true } | { timedOut: false; value: T }> {
    let timer: ReturnType<typeof setTimeout>
    const onTimeout = new Promise<{ timedOut: true }>((resolve) => {
      timer = setTimeout(() => resolve({ timedOut: true }), ms)
    })
    const onSettle = p.then(
      (value) => {
        clearTimeout(timer)
        return { timedOut: false, value }
      },
      (err) => {
        clearTimeout(timer)
        throw err
      },
    )
    return Promise.race([onSettle, onTimeout])
  }

  async function load() {
    if (inFlight.current) return
    inFlight.current = true
    const controller = new AbortController()
    controllerRef.current = controller
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
    try {
      const meResult = await withTimeout(getMe(false, controller.signal), GETME_TIMEOUT_MS)
      if (meResult.timedOut) {
        controller.abort()
        if (mountedRef.current) setError('Analytics unavailable: identity check timed out')
        return
      }
      const user = meResult.value
      if (!user) {
        redirectToLogin('/analytics/dashboard')
        return
      }
      if (!mountedRef.current) return
      setMe(user)
      // Non-admins get the forbidden panel; the API is the authoritative gate, so hiding data here is UX only.
      if (user.role !== 'admin') {
        if (mountedRef.current) setForbidden(true)
        return
      }
      const [sRes, cRes] = await Promise.all([
        fetch(`${API_BASE}/analytics/summary`, authRequestInit({ signal: controller.signal })),
        fetch(`${API_BASE}/analytics/chat`, authRequestInit({ signal: controller.signal })),
      ])
      if (sRes.status === 401 || cRes.status === 401) {
        redirectToLogin('/analytics/dashboard')
        return
      }
      if (!mountedRef.current) return
      const [sFeed, cFeed] = await Promise.all([
        readFeed<Summary>(sRes, 'Search analytics'),
        readFeed<ChatStats>(cRes, 'Chat analytics'),
      ])
      if (!mountedRef.current) return
      setSummary('data' in sFeed ? sFeed.data : null)
      setChat('data' in cFeed ? cFeed.data : null)
      setDegraded({
        summary: 'degraded' in sFeed ? sFeed.degraded : null,
        chat: 'degraded' in cFeed ? cFeed.degraded : null,
      })
      setError('')
      if ('degraded' in sFeed || 'degraded' in cFeed) {
        setUpdated('Unavailable — live figures are missing, not zero')
      } else {
        setUpdated(`Updated ${formatClockTime(Date.now())}`)
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      if (mountedRef.current) setError(`Analytics unavailable: ${(e as Error).message}`)
    } finally {
      clearTimeout(timeout)
      inFlight.current = false
      controllerRef.current = null
    }
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 30000)
    return () => {
      mountedRef.current = false
      clearInterval(t)
      controllerRef.current?.abort()
    }
  }, [])


  const d = summary
  const s = chat

  return (
    <div className="dash-wrap">
      <TopBar me={me} />
      <div className="dash-main">
        <div className="dash-updated">{updated}</div>
        {forbidden ? (
          <div className="dash-error">
            You don&apos;t have access to analytics.
            <div className="dash-error-hint">This dashboard is for administrators only.</div>
          </div>
        ) : null}
        {error ? (
          <div className="dash-error">
            {error}
            <div className="dash-error-hint">Admin access required. Sign in with an admin account to view analytics.</div>
          </div>
        ) : null}
        {degraded.summary || degraded.chat ? (
          <div className="dash-error" role="status">
            Analytics unavailable — some feeds could not be read.
            <div className="dash-error-hint">
              Unavailable: {[degraded.summary, degraded.chat].filter(Boolean).join('; ')}. Their
              figures are missing, not zero — nothing is shown for a feed that failed to load.
            </div>
          </div>
        ) : null}

        {degraded.summary ? (
          <div className="dash-panel">
            <h2>Search analytics (unavailable)</h2>
            <div className="dash-empty">
              Unavailable — the analytics store could not be read. These figures are not zero.
            </div>
          </div>
        ) : null}

        {d ? (
          <>
            <div className="dash-cards">
              <Card label="Searches today" value={fmt(d.searches_today)} hint={`all time: ${fmt(d.searches_total)}`} />
              <Card label="Zero-result rate" value={pct(d.zero_result_rate)} hint="queries that found nothing" warn={d.zero_result_rate > 15} />
              <Card label="Weak-result rate" value={pct(d.weak_result_rate)} hint="below relevance threshold" warn={d.weak_result_rate > 25} />
              <Card label="Avg latency" value={`${d.avg_latency_ms} ms`} hint="server round-trip" />
              <Card label="Cache hit" value={pct(d.cache_hit_rate)} hint="of searches served from cache" />
              <Card label="Filtered" value={pct(d.filtered_rate)} hint="searches with facet/date filters" />
              <Card label="Clicks" value={fmt(d.clicks_total)} hint="results opened by users" />
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Top queries</h2>
                <TopTable rows={d.top_queries} />
              </div>
              <div className="dash-panel">
                <h2>Clicks</h2>
                {d.clicks_total > 0 ? (
                  <>
                    <h2 className="dash-subh">Clicks by position</h2>
                    <table>
                      <thead>
                        <tr>
                          <th scope="col">Result slot</th>
                          <th scope="col" className="num">Clicks</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(d.click_positions)
                          .sort(([a], [b]) => Number(a) - Number(b))
                          .map(([k, n]) => (
                            <tr key={k}>
                              <td>Position {k}</td>
                              <td className="num">{fmt(n)}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                    <h2 className="dash-subh">Most-clicked queries</h2>
                    <TopTable rows={d.click_top_queries} />
                  </>
                ) : (
                  <div className="dash-empty">No clicks recorded yet. Click a result link to start tracking.</div>
                )}
              </div>
            </div>
          </>
        ) : null}

        {s ? (
          <>
            <h2 className="dash-section">Chat usage</h2>
            <div className="dash-cards">
              <Card label="Chat users" value={fmt(s.users)} hint="distinct accounts" />
              <Card label="Conversations" value={fmt(s.sessions)} hint={`today: ${fmt(s.sessions_today)}`} />
              <Card label="Messages" value={fmt(s.messages)} hint="user + assistant" />
              <Card label="Total tokens" value={fmt(s.total_tokens)} hint="prompt + completion" />
              <Card label="Total cost" value={formatCost(s.total_cost)} hint="across all conversations" warn={s.total_cost > 0} />
              <Card label="Avg latency" value={`${s.avg_latency_ms} ms`} hint="per assistant reply" />
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Conversations by cost</h2>
                <ChatTable rows={s.top_by_cost} cost />
              </div>
              <div className="dash-panel">
                <h2>Conversations by tokens</h2>
                <ChatTable rows={s.top_by_tokens} cost={false} />
              </div>
            </div>
          </>
        ) : null}

        {degraded.chat ? (
          <div className="dash-panel">
            <h2 className="dash-section">Chat usage (unavailable)</h2>
            <div className="dash-empty">
              Unavailable — the chat store could not be read. These figures are not zero.
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}