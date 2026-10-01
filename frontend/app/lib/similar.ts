import { API_BASE, authRequestInit } from './auth'
import {
  createDeadline,
  RECOMMEND_DEADLINE_MS,
  RequestTimeoutError,
  type Deadline,
} from './deadline'

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

class BatchRequestError extends Error {
  readonly status: number

  constructor(status: number) {
    super(`Failed to load (${status})`)
    this.status = status
  }
}

const CLIENT_TTL_MS = 5 * 60 * 1000

const MAX_BATCH_IDS = 20

const CLIENT_CACHE_MAX_ENTRIES = 200

const cache = new Map<string, CacheEntry>()
// Single-flight and waiter maps are keyed by cache key: a remount mid-request joins the running one.
const inFlight = new Map<string, Promise<SimilarArticle[]>>()
const waiting = new Map<string, Waiter>()
let flushScheduled = false

/** Bounded two ways — sweep on write, then evict the oldest — because either bound alone leaks. */
function remember(key: string, articles: SimilarArticle[], expiresAt: number): void {
  if (!cache.has(key) && cache.size >= CLIENT_CACHE_MAX_ENTRIES) {
    const now = Date.now()
    for (const [stale, entry] of cache) {
      if (cache.size < CLIENT_CACHE_MAX_ENTRIES) break
      if (entry.expiresAt > now) break
      cache.delete(stale)
    }
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

export function peekSimilarArticles(
  articleId: number | string,
  limit: number
): SimilarArticle[] | null {
  return readCache(cacheKey(articleId, limit))
}

function isIndexable(articleId: number | string): articleId is number {
  if (typeof articleId === 'number') return Number.isInteger(articleId) && articleId > 0
  return /^\d+$/.test(articleId)
}

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

  const chunks: number[][] = []
  for (const articleId of ids) {
    const open = chunks[chunks.length - 1]
    if (open && open.length < MAX_BATCH_IDS) open.push(articleId)
    else chunks.push([articleId])
  }
  // No caller signal on purpose: the request is shared by the whole view, so only the deadline may abort it.
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
      if (list.length) remember(key, list, expiresAt)
    }
    settle(waiters, articles)
  } catch (error) {
    for (const waiter of waiters) inFlight.delete(cacheKey(waiter.articleId, waiter.limit))
    for (const waiter of waiters) {
      waiter.reject(
        deadline.timedOut() ? new RequestTimeoutError(RECOMMEND_DEADLINE_MS) : error
      )
    }
  } finally {
    deadline.clear()
  }
}

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
    waiter.resolve(articles.get(key) ?? [])
  }
}
