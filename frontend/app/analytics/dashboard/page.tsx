'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { API_BASE, authHeaders, getMe, getToken, redirectToLogin } from '../../lib/auth'

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

interface ChatStats {
  sessions: number
  users: number
  messages: number
  total_tokens: number
  total_cost: number
  avg_latency_ms: number
  sessions_today: number
  top_by_cost: [string, number, number, number][]
  top_by_tokens: [string, number, number, number][]
}

function fmt(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? '0' : Number(n).toLocaleString()
}

function inr(v: number | null | undefined): string {
  return v == null ? '₹0' : '₹' + Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

function pct(v: number | null | undefined): string {
  return v == null ? '0%' : `${v}%`
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
          <th scope="col">Query</th>
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
  rows: [string, number, number, number][] | undefined
  cost: boolean
}) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No chat activity yet.</div>
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Conversation</th>
          <th scope="col" className="num">Msgs</th>
          <th scope="col" className="num">{cost ? 'Cost' : 'Tokens'}</th>
          <th scope="col">Updated</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([title, msgs, value, ts]) => (
          <tr key={title + ts}>
            <td>{title}</td>
            <td className="num">{fmt(msgs)}</td>
            <td className="num">{cost ? inr(value) : fmt(value)}</td>
            <td>{new Date(ts * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</td>
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

  async function load() {
    if (!getToken()) {
      redirectToLogin('/analytics/dashboard')
      return
    }
    try {
      const me = await getMe()
      if (!me) {
        redirectToLogin('/analytics/dashboard')
        return
      }
      // Client-side admin gate is a UX convenience only. Authoritative
      // enforcement happens in the backend API (which rejects non-admin
      // requests), so this check can never be the source of truth.
      if (me.role !== 'admin') {
        setForbidden(true)
        return
      }
      const [sRes, cRes] = await Promise.all([
        fetch(`${API_BASE}/analytics/summary`, { headers: authHeaders() }),
        fetch(`${API_BASE}/analytics/chat`, { headers: authHeaders() }),
      ])
      if (sRes.status === 401 || cRes.status === 401) {
        redirectToLogin('/analytics/dashboard')
        return
      }
      if (!sRes.ok) throw new Error(`summary returned HTTP ${sRes.status}`)
      if (!cRes.ok) throw new Error(`chat returned HTTP ${cRes.status}`)
      setSummary((await sRes.json()) as Summary)
      setChat((await cRes.json()) as ChatStats)
      setError('')
      setUpdated(`Updated ${new Date().toLocaleTimeString()}`)
    } catch (e) {
      setError(`Analytics unavailable: ${(e as Error).message}`)
    }
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [])

  function logout() {
    fetch(`${API_BASE}/api/auth/logout`, { method: 'POST', headers: authHeaders() }).catch(() => {})
    redirectToLogin()
  }

  const d = summary
  const s = chat

  return (
    <div className="dash-wrap">
      <header className="dash-topbar">
        <div className="dash-brand">
          VCCircle <span className="dash-dot">·</span> ASK — Analytics
        </div>
        <div className="dash-topbar-links">
          <Link href="/" className="dash-link">
            ← Back to search
          </Link>
          <button type="button" className="dash-logout" onClick={logout}>
            Log out
          </button>
        </div>
      </header>
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
                          .slice(0, 10)
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
              <Card label="Total cost" value={inr(s.total_cost)} hint="across all conversations" warn={s.total_cost > 0} />
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
      </div>
    </div>
  )
}