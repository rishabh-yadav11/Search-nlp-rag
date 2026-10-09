'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import SafeArticleLink from '../components/SafeArticleLink'
import TopBar from '../components/TopBar'
import { API_BASE, authRequestInit, redirectToLogin } from '../lib/auth'
import { formatArticleDate } from '../lib/format'
import { createDeadline } from '../lib/deadline'
import styles from './page.module.css'

// ---- backend feed contract -------------------------------------------------
// GET /api/feed/subscriptions → { subscriptions:[{kind,value,created_at}] }
// POST /api/feed/subscriptions body {kind,value} → {ok,added}
// DELETE /api/feed/subscriptions body {kind,value} → {ok,removed}
// GET /api/feed?limit=N → { results:[SourceSummary], note: str|null }
// GET /facets → { industry:[...], dealtype:[...], tags:[...] }

type SubscriptionKind = 'tag' | 'industry' | 'dealtype'

interface Subscription {
  kind: SubscriptionKind
  value: string
  created_at: number
}

interface FacetOptions {
  industry: string[]
  dealtype: string[]
  tags: string[]
}

interface Article {
  id: number | string
  title: string
  url: string
  published_date?: string
  category?: string
  summary?: string
  author_names?: string[]
  industry_names?: string[]
  dealtype_names?: string[]
  tag_names?: string[]
  content_type?: string
  score?: number
}

const KINDS: SubscriptionKind[] = ['tag', 'industry', 'dealtype']

const KIND_LABELS: Record<SubscriptionKind, string> = {
  tag: 'Tag',
  industry: 'Industry',
  dealtype: 'Deal type',
}

// Which /facets key feeds each kind's datalist.
const FACET_KEY: Record<SubscriptionKind, keyof FacetOptions> = {
  tag: 'tags',
  industry: 'industry',
  dealtype: 'dealtype',
}

/** Empty facets response guard. */
const EMPTY_FACETS: FacetOptions = { industry: [], dealtype: [], tags: [] }

/** Views of the page that are worth a bound, so a hung socket turns into an error the user can act on. */
const FEED_DEADLINE_MS = 15_000
const SUBS_DEADLINE_MS = 15_000
const FACETS_DEADLINE_MS = 10_000
const ACTION_DEADLINE_MS = 10_000
const FEED_LIMIT = 20

export default function FeedPage() {
  const [articles, setArticles] = useState<Article[]>([])
  const [note, setNote] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [subscriptions, setSubscriptions] = useState<Subscription[]>([])
  const [subsError, setSubsError] = useState<string | null>(null)
  const [facets, setFacets] = useState<FacetOptions>(EMPTY_FACETS)
  const [addValues, setAddValues] = useState<Record<SubscriptionKind, string>>({
    tag: '',
    industry: '',
    dealtype: '',
  })
  const [addError, setAddError] = useState<Record<SubscriptionKind, string | null>>({
    tag: null,
    industry: null,
    dealtype: null,
  })
  const [busy, setBusy] = useState<Record<SubscriptionKind, boolean>>({
    tag: false,
    industry: false,
    dealtype: false,
  })

  // Aborting a still-in-flight feed load whenever a newer one starts (e.g. a
  // subscription change while the previous load is pending) keeps the page from
  // letting a stale response overwrite a fresh one.
  const feedControllerRef = useRef<AbortController | null>(null)

  // ---- feed grid -----------------------------------------------------------
  const loadFeed = useCallback(async () => {
    feedControllerRef.current?.abort()
    const controller = new AbortController()
    feedControllerRef.current = controller
    const deadline = createDeadline(FEED_DEADLINE_MS, controller.signal)

    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/api/feed?limit=${FEED_LIMIT}`, authRequestInit({ signal: deadline.signal }))
      if (controller.signal.aborted) return
      if (res.status === 401) {
        redirectToLogin('/feed')
        return
      }
      if (!res.ok) throw new Error(`Failed to load feed: ${res.status}`)
      const data = await res.json()
      if (controller.signal.aborted) return
      setArticles(Array.isArray(data.results) ? data.results : [])
      setNote(typeof data.note === 'string' ? data.note : null)
      setLoading(false)
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(
          deadline.timedOut()
            ? 'The feed did not respond in time. Please try again.'
            : err instanceof Error
              ? err.message
              : 'Failed to load feed'
        )
        setLoading(false)
      }
    } finally {
      deadline.clear()
    }
  }, [])

  // ---- subscription list ---------------------------------------------------
  const loadSubscriptions = useCallback(async () => {
    const deadline = createDeadline(SUBS_DEADLINE_MS)
    try {
      const res = await fetch(`${API_BASE}/api/feed/subscriptions`, authRequestInit({ signal: deadline.signal }))
      if (res.status === 401) {
        redirectToLogin('/feed')
        return
      }
      if (!res.ok) throw new Error(`Failed to load subscriptions: ${res.status}`)
      const data = await res.json()
      setSubscriptions(Array.isArray(data.subscriptions) ? data.subscriptions : [])
      setSubsError(null)
    } catch (err) {
      setSubsError(
        deadline.timedOut()
          ? 'Subscriptions did not load in time. Please try again.'
          : err instanceof Error
            ? err.message
            : 'Failed to load subscriptions'
      )
    } finally {
      deadline.clear()
    }
  }, [])

  // ---- autocomplete vocab --------------------------------------------------
  useEffect(() => {
    const deadline = createDeadline(FACETS_DEADLINE_MS)
    let live = true
    fetch(`${API_BASE}/facets`, authRequestInit({ signal: deadline.signal }))
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!live || !data) return
        setFacets({
          industry: Array.isArray(data.industry) ? data.industry : [],
          dealtype: Array.isArray(data.dealtype) ? data.dealtype : [],
          tags: Array.isArray(data.tags) ? data.tags : [],
        })
      })
      .catch(() => {})
      .finally(() => deadline.clear())
    return () => {
      live = false
      deadline.clear()
    }
  }, [])

  // First load: subscriptions guard the page's auth, and the feed fills in once
  // the backend responds.
  useEffect(() => {
    void loadSubscriptions()
    void loadFeed()
    return () => feedControllerRef.current?.abort()
  }, [loadSubscriptions, loadFeed])

  // ---- actions -------------------------------------------------------------
  async function addSubscription(kind: SubscriptionKind) {
    const value = addValues[kind].trim()
    if (!value) {
      setAddError((e) => ({ ...e, [kind]: 'Enter a value to subscribe to.' }))
      return
    }
    setAddError((e) => ({ ...e, [kind]: null }))
    setBusy((b) => ({ ...b, [kind]: true }))
    const deadline = createDeadline(ACTION_DEADLINE_MS)
    try {
      const res = await fetch(
        `${API_BASE}/api/feed/subscriptions`,
        authRequestInit({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind, value }),
          signal: deadline.signal,
        })
      )
      if (res.status === 401) {
        redirectToLogin('/feed')
        return
      }
      if (!res.ok) throw new Error(`Failed to add subscription: ${res.status}`)
      setAddValues((v) => ({ ...v, [kind]: '' }))
      // The subscription list is the single source of truth from the backend.
      await Promise.all([loadSubscriptions(), loadFeed()])
    } catch (err) {
      if (!deadline.timedOut()) {
        setAddError((e) => ({
          ...e,
          [kind]: err instanceof Error ? err.message : 'Failed to add subscription',
        }))
      }
    } finally {
      deadline.clear()
      setBusy((b) => ({ ...b, [kind]: false }))
    }
  }

  async function removeSubscription(kind: SubscriptionKind, value: string) {
    setAddError((e) => ({ ...e, [kind]: null }))
    setBusy((b) => ({ ...b, [kind]: true }))
    const deadline = createDeadline(ACTION_DEADLINE_MS)
    try {
      const res = await fetch(
        `${API_BASE}/api/feed/subscriptions`,
        authRequestInit({
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind, value }),
          signal: deadline.signal,
        })
      )
      if (res.status === 401) {
        redirectToLogin('/feed')
        return
      }
      if (!res.ok) throw new Error(`Failed to remove subscription: ${res.status}`)
      await Promise.all([loadSubscriptions(), loadFeed()])
    } catch (err) {
      if (!deadline.timedOut()) {
        setAddError((e) => ({
          ...e,
          [kind]: err instanceof Error ? err.message : 'Failed to remove subscription',
        }))
      }
    } finally {
      deadline.clear()
      setBusy((b) => ({ ...b, [kind]: false }))
    }
  }

  return (
    <>
      <TopBar />
      <div className={styles['feed-page']}>
        <div className={styles['feed-header']}>
          <h1>Feed</h1>
          <p className={styles['feed-subtitle']}>
            Follow tags, industries, and deal types to build a personal feed of the stories that matter to you.
          </p>
        </div>

        <section className={styles['subscriptions-card']} aria-label="Subscription manager">
          <h2 className={styles['subscriptions-title']}>Your subscriptions</h2>
          {subsError ? <div className={styles.error}>{subsError}</div> : null}

          <div className={styles['subscription-kinds']}>
            {KINDS.map((kind) => {
              const label = KIND_LABELS[kind]
              const current = subscriptions.filter((s) => s.kind === kind)
              return (
                <div key={kind} className={styles['subscription-kind']}>
                  <div className={styles['subscription-add']}>
                    <input
                      type="text"
                      id={`sub-input-${kind}`}
                      list={`sub-options-${kind}`}
                      placeholder={`Add a ${label.toLowerCase()}…`}
                      value={addValues[kind]}
                      disabled={busy[kind]}
                      onChange={(e) => setAddValues((v) => ({ ...v, [kind]: e.target.value }))}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault()
                          void addSubscription(kind)
                        }
                      }}
                      aria-label={`Add ${label.toLowerCase()}`}
                    />
                    <datalist id={`sub-options-${kind}`}>
                      {facets[FACET_KEY[kind]].map((opt) => (
                        <option key={opt} value={opt} />
                      ))}
                    </datalist>
                    <button
                      type="button"
                      className={styles['add-btn']}
                      disabled={busy[kind]}
                      onClick={() => void addSubscription(kind)}
                    >
                      Add {label}
                    </button>
                  </div>
                  {addError[kind] ? <div className={styles.error}>{addError[kind]}</div> : null}
                  {current.length > 0 ? (
                    <div className={styles['subscription-chips']}>
                      {current.map((s) => (
                        <span key={`${s.kind}:${s.value}`} className={`chip ${styles['subscription-chip']}`}>
                          <span className={styles['chip-label']}>{label}</span>
                          {s.value}
                          <button
                            type="button"
                            className={styles['chip-remove']}
                            disabled={busy[s.kind]}
                            aria-label={`Remove ${label} ${s.value}`}
                            onClick={() => void removeSubscription(s.kind, s.value)}
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className={styles['subscription-empty']}>
                      No {label.toLowerCase()} subscriptions yet.
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </section>

        <h2 className={styles['feed-section-title']}>Your feed</h2>
        {loading ? (
          <div className={styles.loading}>Loading…</div>
        ) : error ? (
          <div className={styles.error}>
            {error}
            <div>
              <button type="button" className={styles.retry} onClick={() => void loadFeed()}>
                Retry
              </button>
            </div>
          </div>
        ) : articles.length === 0 ? (
          <div className={styles.empty}>
            {note ? <p>{note}</p> : null}
            {subscriptions.length === 0 ? (
              <p>You are not following anything yet. Add a tag, industry, or deal type above to start building your feed.</p>
            ) : null}
          </div>
        ) : (
          <div className={styles['articles-grid']}>
            {articles.map((article) => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
        )}
      </div>
    </>
  )
}

function ArticleCard({ article }: { article: Article }) {
  return (
    <div className={styles['article-card']}>
      <SafeArticleLink url={article.url} className={styles['article-link']}>
        <div className={styles['article-content']}>
          <h3 className={styles['article-title']}>{article.title}</h3>
          {article.summary ? <p className={styles['article-summary']}>{article.summary}</p> : null}
          <div className={styles['article-meta']}>
            {article.category ? <span className={styles['article-category']}>{article.category}</span> : null}
            {article.industry_names && article.industry_names.length > 0 ? (
              <span>{article.industry_names.slice(0, 2).join(', ')}</span>
            ) : null}
            {article.dealtype_names && article.dealtype_names.length > 0 ? (
              <span>{article.dealtype_names.slice(0, 2).join(', ')}</span>
            ) : null}
            {article.published_date ? <span>{formatArticleDate(article.published_date)}</span> : null}
          </div>
        </div>
      </SafeArticleLink>
    </div>
  )
}
