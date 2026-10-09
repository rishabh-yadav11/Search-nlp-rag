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
  // NEW (contract §1): kept optional so an admin on an old backend still sees
  // the byte-compatible keys instead of a crash; a missing key renders its
  // panel as an empty state, never as zeroed data.
  latency?: { p50: number; p90: number; p95: number; p99: number }
  hourly_volume?: [string, number][]
  intent_counts?: Record<string, number>
  top_queries_today?: [string, number][]
}

// The first element of each chat row is the OPAQUE session id, never the session
// title, which is the first 60 characters of the user's own question.
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
  // NEW (contract §3). Same optionality convention as Summary.
  latency?: { p50: number; p90: number; p95: number }
  failed_turn_rate?: number
  abandon_rate?: number
  citation_rate?: number
  avg_sources_per_cited?: number
  avg_tokens_per_message?: number
  avg_cost_per_message?: number
  budget?: { daily_limit_usd: number; today_spent_usd: number }
  model_usage?: [string, number, number, number][] // [model, messages, tokens, cost]
  daily_budget?: [string, number][] // [date, costUsd]
  non_llm_answer_rate?: number
}

// GET /analytics/users (contract §2).
type SpenderRow = [userId: string, name: string, totalCostUsd: number]
type AuditRow = [actorId: string, action: string, createdAt: string]

interface UsersStats {
  signups: { today: number; total: number }
  role_distribution: Record<string, number>
  disabled_accounts: number
  active_today: number
  active_last_7d: number
  active_last_30d: number
  last_login_success: number
  signups_14d: [string, number][]
  top_spenders: SpenderRow[]
  audit_recent: AuditRow[]
}

function fmt(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? '0' : Number(n).toLocaleString()
}

function pct(v: number | null | undefined): string {
  return v == null ? '0%' : `${v}%`
}

// A degraded feed is never rendered as data: the backend's counters are legitimately all zero on a quiet day, so a feed that failed to load must be shown as failed.
type Feed<T> = { data: T } | { degraded: string }

/**
 * Classify one analytics response. 503 is the primary signal, but a 200 body carrying `error` is exactly the failure to catch (an intermediary may rewrite the status line).
 */
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

function msLabel(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

/** Hourly volume as a compact bar histogram: one column per hour, height = share of the busiest hour, tooltip = exact count. */
function HourlyVolume({ rows }: { rows: [string, number][] | undefined }) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No searches recorded yet.</div>
  const max = Math.max(1, ...rows.map(([, n]) => n))
  // Simplify the axis label: the backend timestamps look like "2026-10-09T14"
  // (UTC ISO); show the hour segment "14h" below each bar, full bucket in title.
  return (
    <div className="dash-hourly" role="img" aria-label="Searches per hour, last 24 hours">
      {rows.map(([bucket, n]) => {
        // Backend buckets are UTC ISO hour tokens ("2026-10-09T14"); show the
        // hour segment under the bar, or the whole token when it is not one.
        const hour = /T\d{2}$/.test(bucket) ? bucket.slice(-2) : bucket
        return (
          <div key={bucket} className="dash-hour-col">
            <span className="dash-hour-bar" style={{ height: `${Math.max(2, Math.round((60 * n) / max))}px` }} title={`${bucket}:00 — ${n} searches`} />
            <span className="dash-hour-label">{hour}h</span>
          </div>
        )
      })}
    </div>
  )
}

/** A date→value table with the count/cost bar, shared by signups_14d and daily_budget. */
function DateBarTable({ rows, money }: { rows: [string, number][] | undefined; money: boolean }) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No activity yet.</div>
  const max = Math.max(...rows.map(([, v]) => v), 1)
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Day</th>
          <th scope="col" className="num">{money ? 'Cost' : 'Signups'}</th>
          <th scope="col"></th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([date, v]) => (
          <tr key={date}>
            <td>{date}</td>
            <td className="num">{money ? formatCost(v) : fmt(v)}</td>
            <td width="34%">
              <span className="dash-bar" style={{ width: `${Math.round((100 * v) / max)}%` }}></span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function IntentTable({ counts }: { counts: Record<string, number> | undefined }) {
  const entries = Object.entries(counts ?? {})
  if (entries.length === 0) return <div className="dash-empty">No intent data yet.</div>
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Intent class</th>
          <th scope="col" className="num">Searches</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([cls, n]) => (
          <tr key={cls}>
            <td>{cls}</td>
            <td className="num">{fmt(n)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ModelUsageTable({ rows }: { rows: [string, number, number, number][] | undefined }) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No model usage yet.</div>
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col" className="num">Messages</th>
          <th scope="col" className="num">Tokens</th>
          <th scope="col" className="num">Cost</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([model, messages, tokens, cost]) => (
          <tr key={model}>
            <td>{model}</td>
            <td className="num">{fmt(messages)}</td>
            <td className="num">{fmt(tokens)}</td>
            <td className="num">{formatCost(cost)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function SpendersTable({ rows }: { rows: SpenderRow[] | undefined }) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No spend data yet.</div>
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col" className="num">Total cost</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([userId, name, total]) => (
          <tr key={`${userId}-${name}`}>
            <td title={userId}>{name || userId}</td>
            <td className="num">{formatCost(total)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function AuditTable({ rows }: { rows: AuditRow[] | undefined }) {
  if (!rows || rows.length === 0) return <div className="dash-empty">No admin actions recorded yet.</div>
  return (
    <table>
      <thead>
        <tr>
          <th scope="col">Actor</th>
          <th scope="col">Action</th>
          <th scope="col">Time</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([actorId, action, createdAt]) => (
          <tr key={`${actorId}-${createdAt}-${action}`}>
            <td title={actorId}>{actorId.slice(0, 8)}</td>
            <td>{action}</td>
            {/* createdAt may be an epoch number serialized as text or an ISO
                string depending on the slice; handle both for display. */}
            <td>{/^\d+(\.\d+)?$/.test(String(createdAt)) ? formatEpochDateTime(Number(createdAt)) : String(createdAt)}</td>
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
          // Keyed by the session id, never the timestamp: two updates within one poll would remount the row.
          <tr key={sessionId}>
            {/* A non-identifying surrogate: the full id stays in the `title` attribute, and conversation text must never reach this table. */}
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
  const [users, setUsers] = useState<UsersStats | null>(null)
  const [updated, setUpdated] = useState('loading…')
  const [error, setError] = useState('')
  const [forbidden, setForbidden] = useState(false)
  // The signed-in user, shared with the top bar so it does not run a second
  // `/api/auth/me`. Seeded `undefined` so the bar renders no account control
  // until the identity check answers, rather than flashing "Sign in" at an admin.
  const [me, setMe] = useState<AuthUser | null | undefined>(undefined)

  // Which feeds the last poll could not read. A listed feed is NOT rendered as
  // zeros, so a dead store is never mistaken for a quiet day.
  const [degraded, setDegraded] = useState<{
    summary: string | null
    chat: string | null
    users: string | null
  }>({
    summary: null,
    chat: null,
    users: null,
  })

  // Tracks the in-flight load so a polling tick can't race a running load, and
  // so the request can be aborted on unmount.
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
        // Abandoning the promise is not enough: abort so the in-flight
        // `/api/auth/me` socket actually closes.
        controller.abort()
        if (mountedRef.current) setError('Analytics unavailable: identity check timed out')
        return
      }
      const user = meResult.value
      if (!user) {
        // A network failure never reaches here: getMe rethrows it and it is
        // surfaced as an error below, not mistaken for a logout.
        redirectToLogin('/analytics/dashboard')
        return
      }
      if (!mountedRef.current) return
      setMe(user)
      // UX convenience only: the backend API is the authoritative enforcement.
      if (user.role !== 'admin') {
        if (mountedRef.current) setForbidden(true)
        return
      }
      const [sRes, cRes, uRes] = await Promise.all([
        fetch(`${API_BASE}/analytics/summary`, authRequestInit({ signal: controller.signal })),
        fetch(`${API_BASE}/analytics/chat`, authRequestInit({ signal: controller.signal })),
        fetch(`${API_BASE}/analytics/users`, authRequestInit({ signal: controller.signal })),
      ])
      if (sRes.status === 401 || cRes.status === 401 || uRes.status === 401) {
        redirectToLogin('/analytics/dashboard')
        return
      }
      if (!mountedRef.current) return
      const [sFeed, cFeed, uFeed] = await Promise.all([
        readFeed<Summary>(sRes, 'Search analytics'),
        readFeed<ChatStats>(cRes, 'Chat analytics'),
        readFeed<UsersStats>(uRes, 'User analytics'),
      ])
      if (!mountedRef.current) return
      // A feed that failed is dropped, not kept: stale figures must not sit under
      // a fresh-looking "Updated" stamp either.
      setSummary('data' in sFeed ? sFeed.data : null)
      setChat('data' in cFeed ? cFeed.data : null)
      setUsers('data' in uFeed ? uFeed.data : null)
      setDegraded({
        summary: 'degraded' in sFeed ? sFeed.degraded : null,
        chat: 'degraded' in cFeed ? cFeed.degraded : null,
        users: 'degraded' in uFeed ? uFeed.degraded : null,
      })
      setError('')
      if ('degraded' in sFeed || 'degraded' in cFeed || 'degraded' in uFeed) {
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
  const u = users

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
        {degraded.summary || degraded.chat || degraded.users ? (
          <div className="dash-error" role="status">
            Analytics unavailable — some feeds could not be read.
            <div className="dash-error-hint">
              Unavailable: {[degraded.summary, degraded.chat, degraded.users].filter(Boolean).join('; ')}. Their
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
              <Card label="Latency p50" value={d.latency ? msLabel(d.latency.p50) : 'n/a'} hint="median, all searches" />
              <Card label="Latency p90" value={d.latency ? msLabel(d.latency.p90) : 'n/a'} hint="90th percentile" />
              <Card label="Latency p95" value={d.latency ? msLabel(d.latency.p95) : 'n/a'} hint="95th percentile" />
              <Card label="Latency p99" value={d.latency ? msLabel(d.latency.p99) : 'n/a'} hint="99th percentile" />
              <Card label="Cache hit" value={pct(d.cache_hit_rate)} hint="of searches served from cache" />
              <Card label="Filtered" value={pct(d.filtered_rate)} hint="searches with facet/date filters" />
              <Card label="Clicks" value={fmt(d.clicks_total)} hint="results opened by users" />
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Hourly volume</h2>
                <HourlyVolume rows={d.hourly_volume} />
              </div>
              <div className="dash-panel">
                <h2>Searches by intent</h2>
                <IntentTable counts={d.intent_counts} />
              </div>
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Top queries</h2>
                <TopTable rows={d.top_queries_today} />
                <h2 className="dash-subh">All time</h2>
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
                        {/* The backend owns this bound, so don't cap here. */}
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

        {degraded.users ? (
          <div className="dash-panel">
            <h2 className="dash-section">User analytics (unavailable)</h2>
            <div className="dash-empty">
              Unavailable — the user store could not be read. These figures are not zero.
            </div>
          </div>
        ) : null}

        {u ? (
          <>
            <h2 className="dash-section">User analytics</h2>
            <div className="dash-cards">
              <Card label="Signups today" value={fmt(u.signups?.today)} hint={`all time: ${fmt(u.signups?.total)}`} />
              <Card label="Active today" value={fmt(u.active_today)} hint="accounts seen in the last 24h" />
              <Card label="Active 7d" value={fmt(u.active_last_7d)} hint="accounts seen in the last 7 days" />
              <Card label="Active 30d" value={fmt(u.active_last_30d)} hint="accounts seen in the last 30 days" />
              <Card label="Disabled" value={fmt(u.disabled_accounts)} hint="accounts with is_active=0" />
              <Card label="Logins today" value={fmt(u.last_login_success)} hint="successful logins" />
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Role distribution</h2>
                {Object.keys(u.role_distribution ?? {}).length ? (
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Role</th>
                        <th scope="col" className="num">Accounts</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(u.role_distribution ?? {}).map(([role, count]) => (
                        <tr key={role}>
                          <td>{role}</td>
                          <td className="num">{fmt(count)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="dash-empty">No role data yet.</div>
                )}
              </div>
              <div className="dash-panel">
                <h2>Signups — last 14 days</h2>
                <DateBarTable rows={u.signups_14d} money={false} />
              </div>
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Top spenders</h2>
                <SpendersTable rows={u.top_spenders} />
              </div>
              <div className="dash-panel">
                <h2>Recent admin actions</h2>
                <AuditTable rows={u.audit_recent} />
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
              <Card label="Chat latency p50" value={s.latency ? msLabel(s.latency.p50) : 'n/a'} hint="median assistant reply" />
              <Card label="Chat latency p95" value={s.latency ? msLabel(s.latency.p95) : 'n/a'} hint="95th percentile" />
              <Card label="Failed turns" value={s.failed_turn_rate == null ? 'n/a' : pct(s.failed_turn_rate)} hint="aborted / failed attempts" warn={(s.failed_turn_rate ?? 0) > 10} />
              <Card label="Abandon rate" value={s.abandon_rate == null ? 'n/a' : pct(s.abandon_rate)} hint="one-message conversations" />
              <Card label="Citation rate" value={s.citation_rate == null ? 'n/a' : pct(s.citation_rate)} hint="answers that cite sources" />
              <Card label="Avg sources/cited" value={s.avg_sources_per_cited == null ? 'n/a' : s.avg_sources_per_cited.toFixed(1)} hint="sources per cited answer" />
              <Card label="Avg tokens/msg" value={fmt(s.avg_tokens_per_message ?? null)} hint="per message" />
              <Card label="Avg cost/msg" value={s.avg_cost_per_message == null ? 'n/a' : formatCost(s.avg_cost_per_message)} hint="per message" />
              <Card label="Non-LLM answers" value={s.non_llm_answer_rate == null ? 'n/a' : pct(s.non_llm_answer_rate)} hint="smalltalk / fallback answers" />
              <Card
                label="Budget today"
                value={s.budget ? formatCost(s.budget.today_spent_usd) : 'n/a'}
                hint={s.budget ? `daily limit ${formatCost(s.budget.daily_limit_usd)}` : 'no budget configured'}
                warn={Boolean(s.budget && s.budget.daily_limit_usd > 0 && s.budget.today_spent_usd / s.budget.daily_limit_usd >= 0.9)}
              />
            </div>
            <div className="dash-grid2">
              <div className="dash-panel">
                <h2>Model usage</h2>
                <ModelUsageTable rows={s.model_usage} />
              </div>
              <div className="dash-panel">
                <h2>Daily cost — last 14 days</h2>
                <DateBarTable rows={s.daily_budget} money />
              </div>
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