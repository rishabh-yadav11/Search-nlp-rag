import { API_BASE, authRequestInit } from './auth'
import {
  createDeadline,
  RECOMMEND_DEADLINE_MS,
  RequestTimeoutError,
  type Deadline,
} from './deadline'

/**
 * Similar articles for one article, batched across every caller that asks in
 * the same tick.
 *
 * A view renders one `<SimilarArticles>` per result (and chat the same per
 * source), so the cost of a view is one request no matter how many cards it
 * renders, and a module-level cache outlives the component that fetched it.
 * The client TTL is far shorter than the hour the server caches these for:
 * this cache avoids re-fetching a view the user is looking at, it is not a
 * second source of truth.
 */

export interface SimilarArticle {
  id: number | string
  title: string
  url: string
  published_date?: string
  category?: string
  summary?: string
  score?: number
}

interface CacheEntry {
  articles: SimilarArticle[]
  expiresAt: number
}

interface Waiter {
  articleId: number | string
  limit: number
  resolve: (articles: SimilarArticle[]) => void
  reject: (error: unknown) => void
}

/** A batch the server refused, carrying the status so the caller can decide. */
class BatchRequestError extends Error {
  readonly status: number

  constructor(status: number) {
    super(`Failed to load (${status})`)
    this.status = status
  }
}

/** Client-side lifetime for a similar list. The server holds its own copy for an hour. */
const CLIENT_TTL_MS = 5 * 60 * 1000

/**
 * Mirrors the server's SIMILAR_BATCH_MAX_IDS. Because the two copies can drift,
 * `requestBatch` splits and retries rather than trusting this number: a view
 * is answered in more requests instead of going blank.
 */
const MAX_BATCH_IDS = 20

/** Ceiling on cached lists, so a long session cannot grow the cache without bound. */
const CLIENT_CACHE_MAX_ENTRIES = 200

const cache = new Map<string, CacheEntry>()
/** Requests already in flight, so a remount mid-flight joins one instead of starting a second. */
const inFlight = new Map<string, Promise<SimilarArticle[]>>()
/** Callers waiting for the next flush, keyed like the cache. */
const waiting = new Map<string, Waiter>()
let flushScheduled = false

/**
 * Store an answer, making room if the cache is full. Two bounds, because
 * either alone leaks: expired keys are swept on write (a forward-only session
 * never re-reads them), and the ceiling is the backstop for a session that
 * reads faster than it expires.
 */
function remember(key: string, articles: SimilarArticle[], expiresAt: number): void {
  if (!cache.has(key) && cache.size >= CLIENT_CACHE_MAX_ENTRIES) {
    const now = Date.now()
    for (const [stale, entry] of cache) {
      if (cache.size < CLIENT_CACHE_MAX_ENTRIES) break
      // Insertion order is age order, so the first live entry means the rest
      // are live too and there is nothing left worth sweeping.
      if (entry.expiresAt > now) break
      cache.delete(stale)
    }
    // Still full of live entries: the ceiling wins, oldest out.
    if (cache.size >= CLIENT_CACHE_MAX_ENTRIES) {
      const oldest = cache.keys().next()
      if (!oldest.done) cache.delete(oldest.value)
    }
  }
  cache.set(key, { articles, expiresAt })
}

function cacheKey(articleId: number | string, limit: number): string {
  return `${articleId}:${limit}`
}

function readCache(key: string): SimilarArticle[] | null {
  const hit = cache.get(key)
  if (!hit) return null
  if (hit.expiresAt <= Date.now()) {
    cache.delete(key)
    return null
  }
  return hit.articles
}

/**
 * A cache hit, for callers that can render synchronously — a remount inside
 * the TTL paints its list on the first frame instead of flashing a loader.
 */
export function peekSimilarArticles(
  articleId: number | string,
  limit: number
): SimilarArticle[] | null {
  return readCache(cacheKey(articleId, limit))
}

/**
 * The server takes indexed article ids, and `Result.id` is `number | string`.
 * A non-integer id has no vector to search from and would fail validation for
 * the WHOLE batch, so it is answered as empty here instead of being sent.
 */
function isIndexable(articleId: number | string): articleId is number {
  if (typeof articleId === 'number') return Number.isInteger(articleId) && articleId > 0
  return /^\d+$/.test(articleId)
}

/**
 * Similar articles for one article. Callers that ask in the same tick coalesce
 * into a single request.
 */
export function fetchSimilarArticles(
  articleId: number | string,
  limit: number
): Promise<SimilarArticle[]> {
  if (!articleId || !isIndexable(articleId)) return Promise.resolve([])

  const key = cacheKey(articleId, limit)
  const hit = readCache(key)
  if (hit) return Promise.resolve(hit)

  const running = inFlight.get(key)
  if (running) return running

  const promise = new Promise<SimilarArticle[]>((resolve, reject) => {
    waiting.set(key, { articleId, limit, resolve, reject })
  })
  inFlight.set(key, promise)

  if (!flushScheduled) {
    // A macrotask, not a microtask: React runs every effect of a commit
    // synchronously, so by the time this fires all of one view's cards have
    // registered. A microtask would flush between two effects and split the view.
    flushScheduled = true
    setTimeout(flush, 0)
  }
  return promise
}

function flush(): void {
  flushScheduled = false
  const batch = Array.from(waiting.values())
  waiting.clear()
  if (!batch.length) return

  // One request per distinct limit: a view asks for one shape, and grouping
  // by it keeps a stray caller from splitting the rest of its view in two.
  const byLimit = new Map<number, Waiter[]>()
  for (const waiter of batch) {
    const group = byLimit.get(waiter.limit)
    if (group) group.push(waiter)
    else byLimit.set(waiter.limit, [waiter])
  }
  for (const group of byLimit.values()) void send(group)
}

async function send(waiters: Waiter[]): Promise<void> {
  const limit = waiters[0].limit
  const ids: number[] = []
  for (const waiter of waiters) {
    const articleId = waiter.articleId
    if (isIndexable(articleId) && !ids.includes(articleId)) ids.push(articleId)
  }
  if (!ids.length) return settle(waiters, new Map())

  // A view wider than the server's cap is asked for in as many requests as it
  // takes, rather than having its tail quietly dropped: an unanswered id
  // settles as "no similar articles", which looks the same as the truth and is not.
  const chunks: number[][] = []
  for (const articleId of ids) {
    const open = chunks[chunks.length - 1]
    if (open && open.length < MAX_BATCH_IDS) open.push(articleId)
    else chunks.push([articleId])
  }
  // One deadline for the whole batch, armed here rather than in the card: the
  // request is shared by every SimilarArticles in the view, so a backend that
  // accepts the connection and never answers would pin all of them on
  // "Loading..." forever. No caller signal is composed in — one card unmounting
  // must not take the rest of the view's request down — so a timeout is the
  // only thing that can abort this.
  const deadline = createDeadline(RECOMMEND_DEADLINE_MS)
  try {
    const answers = await Promise.all(
      chunks.map((chunk) => fetchChunk(chunk, limit, deadline))
    )
    const byArticle = new Map<string, SimilarArticle[]>()
    for (const answer of answers) {
      for (const [articleId, list] of answer) byArticle.set(articleId, list)
    }
    const expiresAt = Date.now() + CLIENT_TTL_MS
    const articles = new Map<string, SimilarArticle[]>()
    for (const waiter of waiters) {
      const list = byArticle.get(String(waiter.articleId)) ?? []
      const key = cacheKey(waiter.articleId, waiter.limit)
      articles.set(key, list)
      // An empty list is not cached, the same rule the server follows: it is
      // usually a transient miss, and caching it holds a blank card for five minutes.
      if (list.length) remember(key, list, expiresAt)
    }
    settle(waiters, articles)
  } catch (error) {
    // Drop the in-flight entry so a later mount retries instead of replaying this failure.
    for (const waiter of waiters) inFlight.delete(cacheKey(waiter.articleId, waiter.limit))
    // Distinct from a transport failure, so a card can report a timeout, not an opaque AbortError.
    for (const waiter of waiters) {
      waiter.reject(
        deadline.timedOut() ? new RequestTimeoutError(RECOMMEND_DEADLINE_MS) : error
      )
    }
  } finally {
    deadline.clear()
  }
}

/**
 * Ask for one chunk, splitting it if the server will not take it whole.
 *
 * A 422 means the server's cap has drifted below MAX_BATCH_IDS, which is every
 * card in the view failing at once, so the chunk is halved and retried. Every
 * other failure is a real failure and is reported as one: splitting those would
 * turn an outage into a storm of requests.
 */
async function fetchChunk(
  ids: number[],
  limit: number,
  deadline: Deadline
): Promise<Map<string, SimilarArticle[]>> {
  try {
    return await requestBatch(ids, limit, deadline)
  } catch (error) {
    if (!(error instanceof BatchRequestError) || error.status !== 422 || ids.length < 2) {
      throw error
    }
    const middle = Math.ceil(ids.length / 2)
    const [head, tail] = await Promise.all([
      fetchChunk(ids.slice(0, middle), limit, deadline),
      fetchChunk(ids.slice(middle), limit, deadline),
    ])
    for (const [articleId, list] of tail) head.set(articleId, list)
    return head
  }
}

async function requestBatch(
  ids: number[],
  limit: number,
  deadline: Deadline
): Promise<Map<string, SimilarArticle[]>> {
  // The session is an httpOnly cookie, so this POST is credentialed rather than
  // header-bearing: `authRequestInit` attaches the cookie only when API_BASE is
  // a trusted backend, so a runtime-injected base never receives the session.
  const response = await fetch(
    `${API_BASE}/recommend/similar/batch`,
    authRequestInit({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ article_ids: ids, limit }),
      signal: deadline.signal,
    })
  )
  if (!response.ok) throw new BatchRequestError(response.status)
  const data = (await response.json()) as {
    results?: { article_id: number | string; similar_articles?: SimilarArticle[] }[]
  }
  const byArticle = new Map<string, SimilarArticle[]>()
  for (const group of data.results ?? []) {
    byArticle.set(String(group.article_id), group.similar_articles ?? [])
  }
  return byArticle
}

function settle(waiters: Waiter[], articles: Map<string, SimilarArticle[]>): void {
  for (const waiter of waiters) {
    const key = cacheKey(waiter.articleId, waiter.limit)
    inFlight.delete(key)
    // An id the request did not come back for is still answered, as empty, so
    // no caller is left hanging on a promise that never settles.
    waiter.resolve(articles.get(key) ?? [])
  }
}
