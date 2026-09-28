import { API_BASE, authHeaders } from './auth'
import {
  createDeadline,
  RECOMMEND_DEADLINE_MS,
  RequestTimeoutError,
  type Deadline,
} from './deadline'

/**
 * Similar articles for one article, batched across every caller that asks in
 * the same tick (#353).
 *
 * A search view renders one `<SimilarArticles>` per result and the chat
 * sources list does the same per source, so each of those components used to
 * issue its own request for the same view. Eight results meant eight round
 * trips, eight cache reads and eight vector queries to render eight short
 * lists, and because nothing outlived the component, navigating away and back
 * repeated all of it.
 *
 * Two things fix that, and both are here:
 *
 *  - **Coalescing.** Callers that ask within the same tick are answered by one
 *    POST carrying all their ids, so the cost of a view is one request no
 *    matter how many cards it renders.
 *  - **A module-level cache.** The result outlives the component that fetched
 *    it, so a card that remounts renders from memory and issues nothing.
 *
 * The client TTL is deliberately far shorter than the hour the server caches
 * these for: the cache here is about not re-fetching a view the user is
 * looking at, not about being a second source of truth.
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
 * How many ids to put in one request. This mirrors the server's
 * SIMILAR_BATCH_MAX_IDS, which is the one number on the other side of the
 * wire that has to agree -- so `requestBatch` splits and retries rather than
 * trusting it: if the two ever drift, a view is answered in more requests
 * instead of going blank.
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
 * Store an answer, making room if the cache is full.
 *
 * Two bounds, because either alone leaks. Entries expire on read, but a
 * session that only ever moves forward never reads the old ones again, so
 * expired keys are swept when a new entry is written rather than left to
 * pile up. And a sweep cannot help a session that reads faster than it
 * expires, so the ceiling is the backstop: past it, the oldest entry goes
 * regardless of whether it is still fresh.
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
 * A cache hit, for callers that can render synchronously. A card that remounts
 * inside the TTL shows its list on the first paint instead of flashing a
 * loading state for one frame.
 */
export function peekSimilarArticles(
  articleId: number | string,
  limit: number
): SimilarArticle[] | null {
  return readCache(cacheKey(articleId, limit))
}

/**
 * The server takes indexed article ids, and `Result.id` is `number | string`.
 * A result whose id is not a plain integer has no vector to search from, so it
 * is answered as empty here rather than sent along: it would fail validation
 * for the WHOLE batch, and one unsearchable row in a results page would then
 * cost every other row its similar list too.
 */
function isIndexable(articleId: number | string): articleId is number {
  if (typeof articleId === 'number') return Number.isInteger(articleId) && articleId > 0
  return /^\d+$/.test(articleId)
}

/**
 * Similar articles for one article.
 *
 * Concurrent callers coalesce: everything asked for in the same tick goes out
 * as a single request.
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
    // A macrotask rather than a microtask on purpose. React runs every effect
    // of a commit synchronously, so all the cards of one view have registered
    // their waiter by the time this fires and the batch sees all of them. A
    // microtask would flush between two of those effects and split the view.
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

  // A view wider than the server's cap is asked for in as many requests as
  // it takes, rather than having its tail quietly dropped: an unanswered id
  // settles as "no similar articles", which looks the same as the truth and
  // is not. A 25-source chat message is the case this is for.
  const chunks: number[][] = []
  for (const articleId of ids) {
    const open = chunks[chunks.length - 1]
    if (open && open.length < MAX_BATCH_IDS) open.push(articleId)
    else chunks.push([articleId])
  }
  // One deadline for the whole batch, armed here rather than in the card: the
  // request is shared by every SimilarArticles in the view (that is the point
  // of this module), so a backend that accepts the connection and never
  // answers would otherwise pin all of them on "Loading..." forever. No
  // caller signal is composed in — a single card unmounting must not take the
  // rest of the view's request down with it — so a timeout is the only thing
  // that can abort this.
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
      // usually a transient miss, and pinning it client-side for five minutes
      // would hold the blank card long after the data came back.
      if (list.length) remember(key, list, expiresAt)
    }
    settle(waiters, articles)
  } catch (error) {
    // Drop the in-flight entry so a later mount retries rather than replaying
    // this failure from memory for the rest of the session.
    for (const waiter of waiters) inFlight.delete(cacheKey(waiter.articleId, waiter.limit))
    // Callers distinguish this from a transport failure, so a card can say the
    // request ran out of time instead of surfacing an opaque AbortError.
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
 * The cap is a number on the other side of the wire, so the two copies can
 * drift. If the server ever accepts fewer ids than MAX_BATCH_IDS, a 422 is
 * not one card failing -- it is every card in the view failing at once, and
 * the results page goes blank over a configuration detail. Halving and
 * retrying on that one status costs a few extra round trips in a case that
 * should never arise. Every other failure is a real failure and is reported
 * as one: splitting those would turn an outage into a storm of requests.
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
  const response = await fetch(`${API_BASE}/recommend/similar/batch`, {
    method: 'POST',
    headers: authHeaders({ headers: { 'Content-Type': 'application/json' } }),
    body: JSON.stringify({ article_ids: ids, limit }),
    signal: deadline.signal,
  })
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
