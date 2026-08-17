'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

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
}

type Session = {
  id: string
  title: string
  created_at: number
  updated_at: number
  last_preview?: string
}

const API_BASE =
  (typeof window !== 'undefined' && (window as { API_BASE?: string }).API_BASE) ||
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000')

const USER_KEY = 'vccircle_chat_user_id'

function getUserId(): string {
  try {
    const existing = window.localStorage.getItem(USER_KEY)
    if (existing) return existing
    const id = crypto.randomUUID()
    window.localStorage.setItem(USER_KEY, id)
    return id
  } catch {
    return `anon-${Date.now()}`
  }
}

async function api(path: string, init?: RequestInit) {
  const headers = new Headers(init?.headers)
  headers.set('X-User-Id', getUserId())
  if (init?.body) headers.set('Content-Type', 'application/json')
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return res.json() as Promise<Record<string, unknown>>
}

function relativeTime(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null
  return (
    <div className="chat-sources">
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
    </div>
  )
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
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
      const data = await api(`/api/chat/sessions/${sessionId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ content: question }),
      })
      const userMsg = data.user as Message
      const assistantMsg = data.assistant as Message
      setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), userMsg, assistantMsg])
      await loadSessions()
    } catch (err) {
      setMessages((m) => m.filter((x) => x.id !== optimistic.id))
      setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setSending(false)
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
              onClick={() => openSession(s.id)}
            >
              <div className="chat-session-title" title={s.title}>
                {s.title || 'New chat'}
              </div>
              <div className="chat-session-meta">{relativeTime(s.updated_at)}</div>
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
        </div>
      </aside>

      <section className="chat-main">
        <header className="chat-main-head">
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
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      <SourceList sources={m.sources ?? []} />
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {sending ? (
            <div className="chat-msg chat-assistant">
              <div className="chat-msg-bubble">
                <div className="chat-typing">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="chat-composer">
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