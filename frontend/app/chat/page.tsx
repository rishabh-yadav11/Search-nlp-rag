'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import DataViz, { splitContent } from './DataViz'
import TopBar from '../components/TopBar'
import { API_BASE, type AuthUser, authRequestInit, getMe, redirectToLogin } from '../lib/auth'
import { isSafeUrl } from '../lib/safe-url'
import { formatCost, formatEpochRelative } from '../lib/format'
import { createDeadline, CHAT_API_DEADLINE_MS, RequestTimeoutError } from '../lib/deadline'

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

type SessionDetail = Session & {
  truncated?: boolean
  total_messages?: number
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

// Untrusted LLM markdown; `className` stays allowed so citation `<sup class="cite">` markers survive sanitization.
const sanitizeSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    '*': [...(defaultSchema.attributes?.['*'] ?? []), 'className'],
  },
}

// Re-parsing the accumulated markdown on every SSE token is O(n^2), so throttle streaming renders.
const STREAM_RENDER_MS = 50
// Worst-case SSE patience = TIMEOUT × (RETRIES + 1), with linear backoff.
const SSE_TIMEOUT_MS = 45000
const SSE_MAX_RETRIES = 2

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
  const deadline = createDeadline(CHAT_API_DEADLINE_MS, init?.signal ?? null)
  try {
    // The httpOnly session cookie rides on the browser's credentials, not an Authorization header; `authRequestInit` omits `credentials` when API_BASE is untrusted.
    const res = await fetch(
      `${API_BASE}${path}`,
      authRequestInit({ ...init, signal: deadline.signal })
    )
    if (!res.ok) {
      if (res.status === 401) redirectToLogin()
      const detail = await res.text()
      throw new Error(detail || `Request failed (${res.status})`)
    }
    return (await res.json()) as Record<string, unknown>
  } catch (err) {
    if (deadline.timedOut()) {
      throw new RequestTimeoutError(CHAT_API_DEADLINE_MS)
    }
    throw err
  } finally {
    deadline.clear()
  }
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

// Stable identity: a fresh `[]` per render would defeat the `memo` below and re-parse settled markdown.
const NO_SOURCES: Source[] = []

const SourceList = memo(function SourceList({ sources, msg }: { sources: Source[]; msg: Message }) {
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
                  {isSafeUrl(s.url) ? (
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
})

const AnswerBody = memo(function AnswerBody({ content }: { content: string }) {
  const parts = splitContent(content)
  return (
    <>
      {parts.map((part, i) =>
        part.type === 'viz' ? (
          <DataViz key={`viz-${i}`} block={part.block} />
        ) : part.type === 'err' ? (
          <p key={`err-${i}`} className="chat-viz-empty" style={{ margin: '12px 0' }}>
            Chart requested but could not be rendered.
          </p>
        ) : (
          <div key={`md-${i}`} className="markdown-body">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkCitations]}
              rehypePlugins={[rehypeRaw, [rehypeSanitize, sanitizeSchema]]}
            >
              {part.md}
            </ReactMarkdown>
          </div>
        ),
      )}
    </>
  )
})

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [note, setNote] = useState('')
  const [historyTruncated, setHistoryTruncated] = useState<{ hidden: number } | null>(null)
  const [error, setError] = useState('')
  // Seeded `undefined`, not `null`: `null` would flash "Sign in" before `getMe` resolves.
  const [me, setMe] = useState<AuthUser | null | undefined>(undefined)
  // The value is discarded: this exists only to re-run the relative-time format each minute.
  const [, setTick] = useState(0)
  const scrollRef = useRef<HTMLDivElement>(null)
  // Unmount must abort, or the reader keeps streaming into torn-down state.
  const abortRef = useRef<AbortController | null>(null)
  // Tells an intentional cancel apart from a real failure, so a cancel shows no error.
  const cancelledRef = useRef(false)
  // Read through the ref, not the closure, so an in-flight reply to a switched session is discarded.
  const activeIdRef = useRef<string | null>(null)
  // Must be synchronous: `setSending` only lands after two awaits, so an async check would let a double-bill through.
  const sendingRef = useRef(false)

  const loadSessions = useCallback(async () => {
    try {
      const data = await api('/api/chat/sessions')
      const list = Array.isArray(data) ? (data as Session[]) : []
      setSessions(list)
    } catch {
    }
  }, [])

  useEffect(() => {
    // The httpOnly cookie is unreadable by JS, so this is the only sign-in check; a network failure must not redirect.
    getMe()
      .then((user) => {
        setMe(user)
        if (!user) redirectToLogin()
      })
      .catch(() => {
      })
  }, [])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    activeIdRef.current = activeId
  }, [activeId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, sending, streamingContent])

  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 60000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    return () => {
      cancelledRef.current = true
      abortRef.current?.abort()
    }
  }, [])

  const openSession = useCallback(
    async (id: string) => {
      cancelledRef.current = true
      abortRef.current?.abort()
      setActiveId(id)
      setError('')
      setNote('')
      try {
        const data = await api(`/api/chat/sessions/${id}`)
        const msgs = Array.isArray(data.messages) ? (data.messages as Message[]) : []
        const detail = data as SessionDetail
        const total = Number(detail.total_messages ?? msgs.length)
        setHistoryTruncated(
          detail.truncated && total > msgs.length ? { hidden: total - msgs.length } : null
        )
        setMessages(msgs)
      } catch (err) {
        setMessages([])
        setHistoryTruncated(null)
        setError(
          err instanceof RequestTimeoutError
            ? err.message
            : 'Could not load this conversation.'
        )
      }
    },
    []
  )

  const newSession = useCallback(() => {
    cancelledRef.current = true
    abortRef.current?.abort()
    setActiveId(null)
    setMessages([])
    setInput('')
    setError('')
    setNote('')
    setHistoryTruncated(null)
  }, [])

  const send = useCallback(async () => {
    const question = input.trim()
    if (!question || sending || sendingRef.current) return
    sendingRef.current = true
    setError('')
    setNote('')
    cancelledRef.current = false

    let sessionId = activeId
    if (!sessionId) {
      try {
        const created = await api('/api/chat/sessions', { method: 'POST', body: JSON.stringify({}) })
        sessionId = String(created.id)
        setActiveId(sessionId)
        await loadSessions()
      } catch (err) {
        setError(
          err instanceof RequestTimeoutError
            ? err.message
            : 'Could not start a new conversation.'
        )
        sendingRef.current = false
        return
      }
    }

    const optimistic: Message = { id: -Date.now(), role: 'user', content: question, created_at: Date.now() / 1000 }
    setMessages((m) => [...m, optimistic])
    setInput('')
    setSending(true)
    let timedOut = false

    try {
      let res: Response | null = null
      let lastErr: Error | null = null
      let accumulated = ""
      let receivedData = false
      let lastActivity = Date.now()
      // Outlives the retry loop so the read-loop watchdog can abort a silent stream: abort rejects the in-flight `reader.read()`.
      let ctrl: AbortController | null = null
      for (let attempt = 0; attempt <= SSE_MAX_RETRIES; attempt++) {
        if (cancelledRef.current) break
        ctrl = new AbortController()
        abortRef.current = ctrl
        timedOut = false
        lastActivity = Date.now()
        const timer = setInterval(() => {
          if (Date.now() - lastActivity > SSE_TIMEOUT_MS) {
            timedOut = true
            ctrl!.abort()
          }
        }, 5000)
        try {
          res = await fetch(
            `${API_BASE}/api/chat/sessions/${sessionId}/messages/stream`,
            authRequestInit({
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ content: question }),
              signal: ctrl.signal,
            })
          )
          lastErr = null
          break
        } catch (e) {
          lastErr = timedOut
            ? new Error('Connection timed out. Please try again.')
            : e instanceof Error
              ? e
              : new Error('Network error')
          // Our own timeout also aborts this signal, so gate on `timedOut` to spare a user cancel a retry.
          if (ctrl.signal.aborted && !timedOut) break
          if (receivedData) break
          if (attempt < SSE_MAX_RETRIES) {
            await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)))
            if (cancelledRef.current) break
          }
        } finally {
          clearInterval(timer)
        }
      }
      if (!res) {
        throw lastErr ?? new Error('Streaming connection failed')
      }
      if (res.status === 401) redirectToLogin()
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      if (!res.body) throw new Error('Streaming not supported')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let doneMsg: Message | null = null
      let note = ''
      let streamError = ''
      let lastRender = 0

      // Idle watchdog: `lastActivity` is refreshed per chunk and on 'done', so only real silence trips it — before content it aborts, after content it cancels the reader and finalizes what arrived.
      timedOut = false
      lastActivity = Date.now()
      const streamTimer = setInterval(() => {
        if (Date.now() - lastActivity > SSE_TIMEOUT_MS) {
          timedOut = true
          if (!receivedData) {
            ctrl?.abort()
          } else {
            // Cancel rather than abort, so the read loop exits normally instead of throwing.
            reader.cancel().catch(() => {})
          }
        }
      }, 5000)

      try {
        // Extracted so the flush path's in-buffer 'done' dispatches exactly like a streamed one.
        const handleEventBlock = (block: string) => {
          const lines = block.split('\n')
          let type = ''
          let data = ''
          for (const line of lines) {
            if (line.startsWith('event:')) type = line.slice(6).trim()
            else if (line.startsWith('data:')) {
              const value = line.slice(5)
              if (data) data += '\n' + value
              else data = value
            }
          }
          if (!type || !data) return
          try {
            const payload = JSON.parse(data)
            if (type === 'start') {
              const userMsg = payload.user as Message
              setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), userMsg])
            } else if (type === 'delta') {
              const text = payload.text as string
              setStreaming(true)
              receivedData = true
              lastActivity = Date.now()
              accumulated += text
              const now = Date.now()
              if (now - lastRender >= STREAM_RENDER_MS) {
                lastRender = now
                setStreamingContent(accumulated)
              }
            } else if (type === 'done') {
              lastActivity = Date.now()
              doneMsg = payload.message as Message
              note = payload.note ?? ''
            } else if (type === 'error') {
              streamError = payload.error ?? 'Something went wrong.'
            }
          } catch {
          }
        }

        while (true) {
          const { done, value } = await reader.read()
          if (done) {
            // The final 'done' can arrive without its trailing blank line, so parse the leftover buffer before falling back.
            buffer += decoder.decode()
            const remainder = buffer.replace(/\r\n/g, '\n').trim()
            if (remainder) handleEventBlock(remainder)
            break
          }
          lastActivity = Date.now()
          buffer += decoder.decode(value, { stream: true })
          const events = buffer.replace(/\r\n/g, '\n').split('\n\n')
          buffer = events.pop() ?? ''
          for (const evt of events) handleEventBlock(evt)
        }

        // The throttle can drop the last chunk, so force a final render.
        if (accumulated) setStreamingContent(accumulated)

        if (streamError) throw new Error(streamError)
        if (doneMsg) {
          if (activeIdRef.current === sessionId) {
            setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), doneMsg!])
            if (note) setNote(note)
            // The loaded window is a fixed-size tail, and a committed turn adds the user and assistant messages.
            setHistoryTruncated((h) => (h ? { hidden: h.hidden + 2 } : h))
          }
          setStreamingContent('')
          await loadSessions()
        } else if (accumulated) {
          if (activeIdRef.current === sessionId) {
            setMessages((m) => [...m.filter((x) => x.id !== optimistic.id), { id: -Date.now() + 1, role: 'assistant', content: accumulated, created_at: Date.now() / 1000 }])
          }
          setStreamingContent('')
          await loadSessions()
        } else {
          throw new Error('No response received.')
        }
      } finally {
        clearInterval(streamTimer)
        reader.cancel().catch(() => {})
      }
    } catch (err) {
      if (cancelledRef.current) {
        cancelledRef.current = false
        setStreamingContent('')
        return
      }
      setMessages((m) => m.filter((x) => x.id !== optimistic.id))
      setStreamingContent('')
      // A watchdog timeout surfaces a friendly message, not raw AbortError text.
      if (timedOut) {
        setError('Connection timed out. Please try again.')
        return
      }
      setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setSending(false)
      setStreaming(false)
      sendingRef.current = false
    }
  }, [activeId, input, loadSessions, sending])

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await api(`/api/chat/sessions/${id}`, { method: 'DELETE' })
      } catch (err) {
        setError(
          err instanceof RequestTimeoutError
            ? err.message
            : 'Could not delete this conversation.'
        )
      } finally {
        if (id === activeId) newSession()
        await loadSessions()
      }
    },
    [activeId, loadSessions, newSession],
  )

  return (
    <div className="chat-page">
      <TopBar me={me} subtitle={activeId ? 'Conversation' : null} />
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
              >
                <button
                  type="button"
                  className="chat-session-main"
                  onClick={() => openSession(s.id)}
                  aria-label={`Open conversation: ${s.title || 'New chat'}`}
                  style={{
                    border: 'none',
                    background: 'none',
                    padding: 0,
                    margin: 0,
                    font: 'inherit',
                    color: 'inherit',
                    textAlign: 'left',
                    cursor: 'pointer',
                    flex: 1,
                    minWidth: 0,
                  }}
                >
                  <div className="chat-session-title" title={s.title}>
                    {s.title || 'New chat'}
                  </div>
                  <div className="chat-session-meta">
                    <span suppressHydrationWarning>{formatEpochRelative(s.updated_at)}</span>
                    {typeof s.total_cost === 'number' && s.total_cost > 0 ? ` · ${formatCost(s.total_cost)}` : ''}
                  </div>
                </button>
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
        </aside>

        <section className="chat-main">
          <div className="chat-thread" ref={scrollRef} aria-live="polite">
            {historyTruncated ? (
              <div className="chat-note" role="status">
                {historyTruncated.hidden} earlier message{historyTruncated.hidden === 1 ? '' : 's'} not shown —
                this conversation is too long to display in full.
              </div>
            ) : null}
            {messages.length === 0 && !sending ? (
              <div className="chat-empty">
                <h1>Ask VCCircle</h1>
                <p>Ask a question about VCCircle&apos;s news archive. Conversations are saved to your account and kept for 6 months.</p>
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
                        <SourceList sources={m.sources ?? NO_SOURCES} msg={m} />
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
    </div>
  )
}