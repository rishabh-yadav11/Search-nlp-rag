'use client'

import { useEffect, useState } from 'react'
import SafeArticleLink from '../components/SafeArticleLink'
import TopBar from '../components/TopBar'
import { API_BASE, authRequestInit } from '../lib/auth'
import { formatArticleDate } from '../lib/format'
import { createDeadline, RECOMMEND_DEADLINE_MS } from '../lib/deadline'
import type { MouseEvent } from 'react'
import styles from './page.module.css'

interface Article {
  id: number | string
  title: string
  url: string
  published_date?: string
  category?: string
  summary?: string
  industry_names?: string[]
  dealtype_names?: string[]
  score?: number
}

type FeedType = 'personalized' | 'trending' | 'latest'

export default function ForYouPage() {
  const [articles, setArticles] = useState<Article[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [feedType, setFeedType] = useState<FeedType>('personalized')
  const [coldStart, setColdStart] = useState(false)
  const [limit] = useState(20)
  // Bumped by Retry and in the fetch effect's deps, so a retry gets a fresh deadline.
  const [retryCount, setRetryCount] = useState(0)

  useEffect(() => {
    const controller = new AbortController()

    // The deadline aborts the socket on its own; `controller` owns the unmount
    // path, and only that path suppresses the state updates below.
    const deadline = createDeadline(RECOMMEND_DEADLINE_MS, controller.signal)

    const fetchFeed = async () => {
      setLoading(true)
      setError(null)

      try {
        let url: string

        switch (feedType) {
          case 'trending':
            url = `${API_BASE}/recommend/trending?limit=${limit}`
            break
          case 'latest':
            url = `${API_BASE}/recommend/for-you?limit=${limit}`
            break
          case 'personalized':
          default:
            url = `${API_BASE}/recommend/for-you?limit=${limit}`
        }
        const res = await fetch(url, authRequestInit({ signal: deadline.signal }))

        if (!res.ok) {
          throw new Error(`Failed to load feed: ${res.status}`)
        }

        const data = await res.json()

        if (!controller.signal.aborted) {
          setArticles(data.recommendations || data.articles || [])
          setColdStart(data.cold_start || false)
          setLoading(false)
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            deadline.timedOut()
              ? 'The recommendations feed did not respond in time. Please try again.'
              : err instanceof Error
                ? err.message
                : 'Failed to load feed'
          )
          setLoading(false)
        }
      } finally {
        deadline.clear()
      }
    }

    fetchFeed()

    return () => controller.abort()
  }, [feedType, limit, retryCount])

  const handleInteraction = async (articleId: number | string, e: MouseEvent) => {
    // Fire-and-forget but bounded: nothing here would ever release a hung socket.
    // The httpOnly session cookie makes this credentialed, not header-bearing:
    // `authRequestInit` attaches the cookie only when API_BASE is a trusted backend.
    const deadline = createDeadline(RECOMMEND_DEADLINE_MS)
    try {
      await fetch(
        `${API_BASE}/recommend/interaction`,
        authRequestInit({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ article_id: articleId, interaction_type: 'click' }),
          signal: deadline.signal,
        })
      )
    } catch (err) {
      if (!deadline.timedOut()) console.error('Failed to record interaction:', err)
    } finally {
      deadline.clear()
    }
  }

  return (
    <>
      <TopBar />
      <div className={styles['for-you-page']}>
        <div className={styles['for-you-header']}>
          <h1>For You</h1>
          <div className={styles['for-you-tabs']}>
            <button
              type="button"
              className={`${styles.tab} ${feedType === 'personalized' ? styles.active : ''}`}
              onClick={() => setFeedType('personalized')}
            >
              Recommended
            </button>
            <button
              type="button"
              className={`${styles.tab} ${feedType === 'trending' ? styles.active : ''}`}
              onClick={() => setFeedType('trending')}
            >
              Trending
            </button>
            <button
              type="button"
              className={`${styles.tab} ${feedType === 'latest' ? styles.active : ''}`}
              onClick={() => setFeedType('latest')}
            >
              Latest
            </button>
          </div>
        </div>

        {coldStart && (
          <div className={styles['cold-start-notice']}>
            You have not interacted with any articles yet. We are showing the latest stories.
            Start clicking on articles to get personalized recommendations!
          </div>
        )}

        {loading ? (
          <div className={styles.loading}>Loading...</div>
        ) : error ? (
          <div className={styles.error}>
            {error}
            <div>
              <button type="button" className={styles.retry} onClick={() => setRetryCount((n) => n + 1)}>
                Retry
              </button>
            </div>
          </div>
        ) : articles.length === 0 ? (
          <div className={styles.empty}>No articles found.</div>
        ) : (
          <div className={styles['articles-grid']}>
            {articles.map((article) => (
              <ArticleCard
                key={article.id}
                article={article}
                onInteraction={(e) => handleInteraction(article.id, e)}
              />
            ))}
          </div>
        )}
      </div>
    </>
  )
}

function ArticleCard({
  article,
  onInteraction,
}: {
  article: Article
  onInteraction: (e: MouseEvent) => void
}) {
  return (
    <div className={styles['article-card']} onClick={onInteraction}>
      <SafeArticleLink url={article.url} className={styles['article-link']}>
        <div className={styles['article-content']}>
          <h2 className={styles['article-title']}>{article.title}</h2>
          {article.summary && (
            <p className={styles['article-summary']}>{article.summary}</p>
          )}
          <div className={styles['article-meta']}>
            {article.category && (
                <span className={styles['article-category']}>{article.category}</span>
            )}
            {article.industry_names && article.industry_names.length > 0 && (
              <span>
                {article.industry_names.slice(0, 2).join(', ')}
              </span>
            )}
            {article.published_date && (
              <span>
                {formatArticleDate(article.published_date)}
              </span>
            )}
          </div>
        </div>
      </SafeArticleLink>
    </div>
  )
}
