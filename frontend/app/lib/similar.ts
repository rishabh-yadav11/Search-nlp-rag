import { API_BASE, authHeaders } from './auth'

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
  reject: (error: Error) => void
}

/** Client-side lifetime for a similar list. The server holds its own copy for an hour. */
const CLIENT_TTL_MS = 5 * 60 * 1000

/** The most ids the server accepts in one batch (SIMILAR_BATCH_MAX_IDS). */
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
 * Store an answer, dropping the oldest entry when the cache is full.
 *
 * Entries expire on read, but a session that only ever moves forward would
 * never read the old ones, so without a ceiling this map would hold every
 * article anyone searched for until the tab closed.
 */
function remember(key: string, articles: SimilarArticle[], expiresAt: number): void {
  if (!cache.has(key) && cache.size >= CLIENT_CACHE_MAX_ENTRIES) {
    const oldest = cache.keys().next()
    if (!oldest.done) cache.delete(oldest.value)
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

  try {
    const answers = await Promise.all(chunks.map((chunk) => requestBatch(chunk, limit)))
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
    for (const waiter of waiters) waiter.reject(error as Error)
  }
}

async function requestBatch(
  ids: number[],
  limit: number
): Promise<Map<string, SimilarArticle[]>> {
  const response = await fetch(`${API_BASE}/recommend/similar/batch`, {
    method: 'POST',
    headers: authHeaders({ headers: { 'Content-Type': 'application/json' } }),
    body: JSON.stringify({ article_ids: ids, limit }),
  })
  if (!response.ok) throw new Error('Failed to load')
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
