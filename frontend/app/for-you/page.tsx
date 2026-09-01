'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { API_BASE, authHeaders, getToken } from '../lib/auth'
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

  useEffect(() => {
    const controller = new AbortController()

    const fetchFeed = async () => {
      setLoading(true)
      setError(null)

      try {
        let url: string
        const headers = authHeaders()

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

        const res = await fetch(url, {
          signal: controller.signal,
          headers,
        })

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
          setError(err instanceof Error ? err.message : 'Failed to load feed')
          setLoading(false)
        }
      }
    }

    fetchFeed()

    return () => controller.abort()
  }, [feedType, limit])

  const handleInteraction = async (articleId: number | string, e: MouseEvent) => {
    const token = getToken()
    if (!token) return

    try {
      await fetch(`${API_BASE}/recommend/interaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ article_id: articleId, interaction_type: 'click' }),
      })
    } catch (err) {
      console.error('Failed to record interaction:', err)
    }
  }

  return (
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
        <div className={styles.error}>{error}</div>
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
      <Link href={article.url} target="_blank" rel="noopener noreferrer" className={styles['article-link']}>
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
                {new Date(article.published_date).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>
      </Link>
    </div>
  )
}
