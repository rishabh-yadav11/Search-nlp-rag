'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import remarkGfm from 'remark-gfm'
import DataViz, { splitContent } from './DataViz'
import { API_BASE, authHeaders, getToken, redirectToLogin } from '../lib/auth'

type Source = {
  id: number
  title: string
  url: string
  published_date?: string | null
  category?: string | null
  summary?: string
  score: number
}

type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  created_at: number
  prompt_tokens?: number
  completion_tokens?: number
  cost?: number
  latency_ms?: number
}

type Session = {
  id: string
  title: string
  created_at: number
  updated_at: number
  last_preview?: string
  total_cost?: number
}

const CITATION_RE = /\[\d+\]/

// Re-rendering the accumulated markdown on every SSE token is O(n^2): each
// token re-parses the entire answer so far. Throttle streaming renders to at
// most this many ms apart; the final chunk is always flushed when the stream
// ends.
const STREAM_RENDER_MS = 50

function remarkCitations() {
  return (tree: any) => {
    walk(tree, (node, parent, index) => {
      if (node.type !== 'text' || !CITATION_RE.test(node.value)) return
      const segments = node.value.split(/(\[\d+\])/g)
      if (segments.length <= 1) return
      const nodes = segments
        .filter((s: string) => s !== '')
        .map((s: string) => (/^\[\d+\]$/.test(s) ? { type: 'html', value: `<sup class="cite">${s}</sup>` } : { type: 'text', value: s }))
      parent.children.splice(index, 1, ...nodes)
    })
  }
}

function walk(node: any, fn: (node: any, parent: any, index: number) => void) {
  if (!node || typeof node !== 'object' || !Array.isArray(node.children)) return
  for (let i = 0; i < node.children.length; i++) {
    const child = node.children[i]
    fn(child, node, i)
    if (child.type === 'text') continue
    walk(child, fn)
  }
}

async function api(path: string, init?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers: authHeaders(init) })
  if (!res.ok) {
    if (res.status === 401) redirectToLogin()
    const detail = await res.text()
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return res.json() as Promise<Record<string, unknown>>
}

// Always format with a fixed locale/timezone so output is identical across
// clients and never diverges on hydration. `now` is computed at call time so
// relative strings keep advancing while the page is open.
function relativeTime(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`
  return new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
}

function formatCost(cost: number): string {
  if (cost <= 0) return ''
  if (cost >= 1) return `₹${cost.toFixed(2)}`
  if (cost >= 0.01) return `₹${cost.toFixed(4)}`
  return `₹${cost.toFixed(6)}`
}

function formatTime(ms: number): string {
  if (ms <= 0) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function UsageLine({ msg }: { msg: Message }) {
  const tokens = (msg.prompt_tokens ?? 0) + (msg.completion_tokens ?? 0)
  const cost = msg.cost ?? 0
  const latency = msg.latency_ms ?? 0
  if (!tokens && !latency) return null
  return (
    <div className="chat-usage">
      {latency ? <span className="chat-usage-time">⏱ {formatTime(latency)}</span> : null}
      <span>{tokens.toLocaleString()} tokens</span>
      {cost > 0 ? <span className="chat-usage-cost">{formatCost(cost)}</span> : null}
    </div>
  )
}

function SourceList({ sources, msg }: { sources: Source[]; msg: Message }) {
  const [open, setOpen] = useState(false)
  if (!sources.length && !((msg.prompt_tokens ?? 0) + (msg.completion_tokens ?? 0))) return null
  return (
    <div className="chat-sources">
      <UsageLine msg={msg} />
      {sources.length ? (
        <>
          <button type="button" className="chat-sources-toggle" onClick={() => setOpen((o) => !o)}>
            {open ? 'Hide' : 'Show'} sources ({sources.length})
          </button>
          {open && (
            <ol className="chat-sources-list">
              {sources.map((s) => (
                <li key={s.id}>
                  {/^https?:\/\//.test(s.url) ? (
                    <a href={s.url} target="_blank" rel="noopener noreferrer">
                      {s.title || `Source ${s.id}`}
                    </a>
                  ) : (
                    <span>{s.title || `Source ${s.id}`}</span>
                  )}
                  {s.published_date ? <span className="chat-sources-date">{s.published_date.slice(0, 10)}</span> : null}
                </li>
              ))}
            </ol>
          )}
        </>
      ) : null}
    </div>
  )
}

function AnswerBody({ content }: { content: string }) {
  const parts = splitContent(content)
  return (
    <>
      {parts.map((part, i) =>
        part.type === 'viz' ? (
          <DataViz key={`viz-${i}`} block={part.block} />
        ) : (
          <ReactMarkdown key={`md-${i}`} remarkPlugins={[remarkGfm, remarkCitations]} rehypePlugins={[rehypeRaw]}>
            {part.md}
          </ReactMarkdown>
        ),
      )}
    </>
  )
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  const loadSessions = useCallback(async () => {
    try {
      const data = await api('/api/chat/sessions')
      const list = Array.isArray(data) ? (data as Session[]) : []
      setSessions(list)
    } catch {
      /* sidebar is best-effort */
    }
  }, [])

  useEffect(() => {
    if (!getToken()) redirectToLogin()
  }, [])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, sending])

  const openSession = useCallback(
    async (id: string) => {
      setActiveId(id)
      setError('')
      try {
        const data = await api(`/api/chat/sessions/${id}`)
        const msgs = Array.isArray(data.messages) ? (data.messages as Message[]) : []
        setMessages(msgs)
      } catch {
        setMessages([])
        setError('Could not load this conversation.')
      }
    },
    [],
  )

  const newSession = useCallback(() => {
    setActiveId(null)
    setMessages([])
    setInput('')
    setError('')
    setNote('')
  }, [])

  const send = useCallback(async () => {
    const question = input.trim()
    if (!question || sending) return
    setError('')

    let sessionId = activeId
    if (!sessionId) {
      try {
        const created = await api('/api/chat/sessions', { method: 'POST', body: JSON.stringify({}) })
        sessionId = String(created.id)
        setActiveId(sessionId)
        await loadSessions()
      } catch {
        setError('Could not start a new conversation.')
        return
      }
    }

    const optimistic: Message = { id: -Date.now(), role: 'user', content: question, created_at: Date.now() / 1000 }
    setMessages((m) => [...m, optimistic])
    setInput('')
    setSending(true)

    try {
      const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}/messages/stream`, {
        method: 'POST',
        headers: authHeaders({ headers: { 'Content-Type': 'application/json' } }),
        body: JSON.stringify({ content: question }),
      })
      if (res.status === 401) redirectToLogin()
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      if (!res.body) throw new Error('Streaming not supported')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulated = ''
      let doneMsg: Message | null = null
      let note = ''
      let streamError = ''
      let lastRender = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() ?? ''
        for (const evt of events) {
          const lines = evt.split('\n')
          let type = ''
          let data = ''
          for (const line of lines) {
            if (line.startsWith('event:')) type = line.slice(6).trim()
            else if (line.startsWith('data:')) data += line.slice(5).trim()
          }
          if (!type || !data) continue
          try {
            const payload = JSON.parse(data)
            if (type === 'start') {
              const userMsg = payload.user as Message
              setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), userMsg])
            } else if (type === 'delta') {
              const text = payload.text as string
              setStreaming(true)
              accumulated += text
              const now = Date.now()
              if (now - lastRender >= STREAM_RENDER_MS) {
                lastRender = now
                setStreamingContent(accumulated)
              }
            } else if (type === 'done') {
              doneMsg = payload.message as Message
              note = payload.note ?? ''
            } else if (type === 'error') {
              streamError = payload.error ?? 'Something went wrong.'
            }
          } catch {
            /* skip malformed event */
          }
        }
      }

      // Flush the final text so the last chunk renders even if it arrived
      // inside the throttle window.
      if (accumulated) setStreamingContent(accumulated)

      if (streamError) throw new Error(streamError)
      if (doneMsg) {
        setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), doneMsg!])
        setStreamingContent('')
        if (note) setNote(note)
        await loadSessions()
      } else if (accumulated) {
        setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), { id: -Date.now() + 1, role: 'assistant', content: accumulated, created_at: Date.now() / 1000 }])
        setStreamingContent('')
        await loadSessions()
      } else {
        throw new Error('No response received.')
      }
    } catch (err) {
      setMessages((m) => m.filter((x) => x.id !== optimistic.id))
      setStreamingContent('')
      setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setSending(false)
      setStreaming(false)
    }
  }, [activeId, input, loadSessions, sending])

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await api(`/api/chat/sessions/${id}`, { method: 'DELETE' })
      } finally {
        if (id === activeId) newSession()
        await loadSessions()
      }
    },
    [activeId, loadSessions, newSession],
  )

  return (
    <div className="chat-app">
      <aside className="chat-sidebar">
        <div className="chat-sidebar-head">
          <button type="button" className="chat-new-btn" onClick={newSession}>
            + New chat
          </button>
        </div>
        <nav className="chat-session-list" aria-label="Conversations">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`chat-session-item${s.id === activeId ? ' active' : ''}`}
              role="button"
              tabIndex={0}
              aria-label={`Open conversation: ${s.title || 'New chat'}`}
              onClick={() => openSession(s.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  openSession(s.id)
                }
              }}
            >
              <div className="chat-session-title" title={s.title}>
                {s.title || 'New chat'}
              </div>
              <div className="chat-session-meta">
                <span suppressHydrationWarning>{relativeTime(s.updated_at)}</span>
                {typeof s.total_cost === 'number' && s.total_cost > 0 ? ` · ${formatCost(s.total_cost)}` : ''}
              </div>
              <button
                type="button"
                className="chat-session-del"
                aria-label="Delete conversation"
                onClick={(e) => {
                  e.stopPropagation()
                  removeSession(s.id)
                }}
              >
                ✕
              </button>
            </div>
          ))}
        </nav>
        <div className="chat-sidebar-foot">
          <Link href="/" className="chat-back-link">
            ← Back to search
          </Link>
          <button
            type="button"
            className="chat-logout"
            onClick={() => {
              fetch(`${API_BASE}/api/auth/logout`, { method: 'POST', headers: authHeaders() }).catch(() => {})
              redirectToLogin()
            }}
          >
            Log out
          </button>
        </div>
      </aside>

      <section className="chat-main">
        <header className="chat-main-head">
          <Link href="/" className="chat-main-brand">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/vccircle-wordmark.svg" alt="VCCircle" width={110} height={30} />
          </Link>
          <span className="chat-main-title">ASK VCCircle</span>
          {activeId ? <span className="chat-main-sub">Conversation</span> : null}
        </header>

        <div className="chat-thread" ref={scrollRef} aria-live="polite">
          {messages.length === 0 && !sending ? (
            <div className="chat-empty">
              <h1>Ask VCCircle</h1>
              <p>Ask a question about VCCircle&apos;s news archive. Conversations are saved to your device and kept for 6 months.</p>
            </div>
          ) : (
            messages.map((m) => (
              <div key={m.id} className={`chat-msg chat-${m.role}`}>
                <div className="chat-msg-bubble">
                  {m.role === 'user' ? (
                    <div className="chat-msg-plain">{m.content}</div>
                  ) : (
                    <div className="chat-msg-answer">
                      <AnswerBody content={m.content} />
                      <SourceList sources={m.sources ?? []} msg={m} />
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {sending && !streaming ? (
            <div className="chat-msg chat-assistant">
              <div className="chat-msg-bubble">
                <div className="chat-typing">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          ) : streamingContent ? (
            <div className="chat-msg chat-assistant">
              <div className="chat-msg-bubble">
                <div className="chat-msg-answer">
                  <AnswerBody content={streamingContent} />
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="chat-composer">
          {note ? (
            <div className="chat-note">
              {note}
            </div>
          ) : null}
          {error ? (
            <div className="chat-error" role="alert">
              {error}
            </div>
          ) : null}
          <form
            onSubmit={(e) => {
              e.preventDefault()
              send()
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="Ask about deals, funding, IPOs, companies…"
              rows={1}
              disabled={sending}
              aria-label="Message"
            />
            <button type="submit" disabled={sending || !input.trim()}>
              Send
            </button>
          </form>
        </div>
      </section>
    </div>
  )
}