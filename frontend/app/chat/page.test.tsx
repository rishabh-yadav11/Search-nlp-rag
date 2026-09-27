/**
 * Issue #269 — streaming renders re-parse every settled assistant answer.
 *
 * `AnswerBody` is the only component that pays for markdown parsing, and it
 * parses from the first statement of its body (`splitContent(content)`). The
 * spy below delegates to the real implementation, so the counts below are an
 * exact measure of how many times each answer was re-parsed while the DOM that
 * follows is genuinely rendered — no synthetic replica of the page is involved.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { TOKEN_KEY } from '../lib/auth'
import type * as DataVizModule from './DataViz'
import ChatPage from './page'

const { splitCalls } = vi.hoisted(() => ({ splitCalls: [] as string[] }))

vi.mock('./DataViz', async () => {
  const actual = await vi.importActual<typeof DataVizModule>('./DataViz')
  return {
    ...actual,
    splitContent: (content: string) => {
      splitCalls.push(content)
      return actual.splitContent(content)
    },
  }
})

const { similarRenders } = vi.hoisted(() => ({ similarRenders: [] as number[] }))

// `SimilarArticles` is rendered by `SourceList` and nothing else, so its
// render count is an exact count of `SourceList` renders for a message whose
// sources are expanded. Stubbed so the measurement needs no network.
vi.mock('../components/SimilarArticles', () => ({
  default: ({ articleId }: { articleId: number }) => {
    similarRenders.push(articleId)
    return <span data-testid={`similar-${articleId}`} />
  },
}))

const SESSION = { id: 's1', title: 'Budget', created_at: 1_700_000_000, updated_at: 1_700_000_100 }

// Two settled turns. The first assistant message deliberately carries NO
// `sources`, which is what makes the `sources={m.sources ?? []}` prop literal
// on the hot render path allocate a fresh array on every tick.
const SETTLED = [
  { id: 1, role: 'user', content: 'question one', created_at: 1_700_000_000 },
  { id: 2, role: 'assistant', content: 'SETTLED_ONE answer', created_at: 1_700_000_010 },
  { id: 3, role: 'user', content: 'question two', created_at: 1_700_000_020 },
  {
    id: 4,
    role: 'assistant',
    content: 'SETTLED_TWO answer',
    created_at: 1_700_000_030,
    sources: [{ id: 900, title: 'Source A', url: 'https://example.com/a', score: 0.5 }],
  },
] as const

const SETTLED_TEXTS = ['SETTLED_ONE answer', 'SETTLED_TWO answer']

// Six deltas, each spaced past the 50 ms render throttle, so every one of them
// is a real streaming render — the ~20/s burst the issue describes.
const DELTAS = ['LIVE a', 'LIVE ab', 'LIVE abc', 'LIVE abcd', 'LIVE abcde', 'LIVE abcdef']
const STREAM_PREFIX = 'LIVE'
const FINAL_ANSWER = 'LIVE abcdef done'

let streamCtrl: ReadableStreamDefaultController<Uint8Array> | null = null

// The fetch stub is used both for JSON endpoints and for the SSE response, so
// its shape is pinned explicitly: without it TS cannot infer the callbacks
// and reports them as implicit `any` (TS7023).
type StubResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
  text: () => Promise<string>
  body?: ReadableStream<Uint8Array>
}

function jsonResponse(data: unknown): StubResponse {
  return { ok: true, status: 200, json: async () => data, text: async () => JSON.stringify(data) }
}

beforeEach(() => {
  splitCalls.length = 0
  similarRenders.length = 0
  streamCtrl = null
  localStorage.setItem(TOKEN_KEY, 'test-token')
  // jsdom implements neither; ChatPage scrolls the thread on every message.
  Element.prototype.scrollTo = function scrollTo() {}
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL): Promise<StubResponse> => {
      const url = String(input)
      if (url.includes('/messages/stream')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({}),
          text: async () => '',
          body: new ReadableStream<Uint8Array>({
            start(controller) {
              streamCtrl = controller
            },
          }),
        }
      }
      if (url.endsWith('/api/chat/sessions')) return jsonResponse([SESSION])
      if (url.includes('/api/chat/sessions/')) return jsonResponse({ ...SESSION, messages: SETTLED })
      return jsonResponse({ similar_articles: [] })
    })
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const countFor = (text: string) => splitCalls.filter((c) => c === text).length

async function openConversation() {
  render(<ChatPage />)
  fireEvent.click(await screen.findByLabelText('Open conversation: Budget'))
  await screen.findByText('SETTLED_ONE answer')
  await screen.findByText('SETTLED_TWO answer')
}

async function streamABurst() {
  const encoder = new TextEncoder()
  fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'third question' } })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  })
  for (const text of DELTAS) {
    await act(async () => {
      streamCtrl!.enqueue(encoder.encode(`event: delta\ndata: ${JSON.stringify({ text })}\n\n`))
      // Longer than the 50 ms throttle, so each delta is its own render.
      const { promise, resolve } = Promise.withResolvers<void>()
      setTimeout(resolve, 60)
      await promise
    })
  }
  await act(async () => {
    streamCtrl!.enqueue(
      encoder.encode(
        `event: done\ndata: ${JSON.stringify({
          message: { id: 9, role: 'assistant', content: FINAL_ANSWER, created_at: 1_700_000_099 },
        })}\n\n`
      )
    )
    streamCtrl!.close()
  })
  await screen.findByText(FINAL_ANSWER)
}

describe('ChatPage — settled answers are not re-parsed while a new answer streams', () => {
  it('leaves the settled render count untouched across a streaming burst', async () => {
    await openConversation()

    const before = SETTLED_TEXTS.map(countFor)
    expect(before).toEqual([1, 1])

    await streamABurst()

    expect(SETTLED_TEXTS.map(countFor)).toEqual(before)
  })

  it('still re-renders the streaming answer on every throttled tick', async () => {
    await openConversation()

    expect(splitCalls.filter((c) => c.startsWith(STREAM_PREFIX)).length).toBe(0)
    await streamABurst()

    // One render per delta that landed outside the throttle window, plus the
    // final flush and the commit into `messages`.
    expect(splitCalls.filter((c) => c.startsWith(STREAM_PREFIX)).length).toBeGreaterThanOrEqual(DELTAS.length)
    expect(countFor(FINAL_ANSWER)).toBeGreaterThanOrEqual(1)
  })

  it('keeps an opened sources list open across a streaming burst', async () => {
    await openConversation()

    fireEvent.click(screen.getByRole('button', { name: /Show sources/ }))
    expect(screen.getByRole('link', { name: 'Source A' })).toBeTruthy()

    await streamABurst()

    expect(screen.getByRole('link', { name: 'Source A' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Hide sources/ })).toBeTruthy()
  })

  it('does not re-render a settled, expanded source list across a streaming burst', async () => {
    await openConversation()

    fireEvent.click(screen.getByRole('button', { name: /Show sources/ }))
    const before = similarRenders.length
    expect(before).toBe(1)

    await streamABurst()

    expect(similarRenders.length).toBe(before)
  })
})

describe('ChatPage — a server-truncated thread is announced, not silently shortened', () => {
  // The server returns only the most recent CHAT_SESSION_MESSAGE_LIMIT messages
  // and flags the rest (issue #258). Dropping older messages without saying so
  // would look like the user's history vanishing.
  function stubSessionDetail(detail: Record<string, unknown>) {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL): Promise<StubResponse> => {
        const url = String(input)
        // The SSE route must be stubbed too, so a turn can be sent against a
        // truncated thread; `streamABurst` writes into the controller it
        // captures.
        if (url.includes('/messages/stream')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({}),
            text: async () => '',
            body: new ReadableStream<Uint8Array>({
              start(controller) {
                streamCtrl = controller
              },
            }),
          }
        }
        if (url.endsWith('/api/chat/sessions')) return jsonResponse([SESSION])
        if (url.includes('/api/chat/sessions/')) return jsonResponse({ ...SESSION, ...detail })
        return jsonResponse({ similar_articles: [] })
      })
    )
  }

  async function openBudget() {
    render(<ChatPage />)
    fireEvent.click(await screen.findByLabelText('Open conversation: Budget'))
    await screen.findByText('SETTLED_TWO answer')
  }

  it('tells the user how many earlier messages were not shown', async () => {
    stubSessionDetail({
      messages: SETTLED,
      truncated: true,
      total_messages: SETTLED.length + 135,
    })

    await openBudget()

    const notice = await screen.findByRole('status')
    expect(notice.textContent).toContain('135')
    expect(notice.textContent?.toLowerCase()).toContain('not shown')
  })

  it('shows no notice when the server returned the whole thread', async () => {
    stubSessionDetail({ messages: SETTLED, truncated: false, total_messages: SETTLED.length })

    await openBudget()

    expect(screen.queryByRole('status')).toBeNull()
  })

  it('shows no notice for a response from a server that does not report truncation', async () => {
    stubSessionDetail({ messages: SETTLED })

    await openBudget()

    expect(screen.queryByRole('status')).toBeNull()
  })

  it('grows the hidden count as further turns are added to the thread', async () => {
    stubSessionDetail({ messages: SETTLED, truncated: true, total_messages: SETTLED.length + 10 })

    await openBudget()
    expect((await screen.findByRole('status')).textContent).toContain('10')

    await streamABurst()

    // One more turn = one more user message and one more answer stored, and
    // the loaded window is a fixed-size tail, so 10 hidden becomes 12.
    expect(screen.getByRole('status').textContent).toContain('12')
  })
})
