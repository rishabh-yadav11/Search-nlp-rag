'use client'

import { useEffect, useState } from 'react'
import SafeArticleLink from './SafeArticleLink'
import { API_BASE, authHeaders } from '../lib/auth'
import { formatArticleDate } from '../lib/format'
import styles from './SimilarArticles.module.css'

interface Article {
  id: number | string
  title: string
  url: string
  published_date?: string
  category?: string
  summary?: string
  score?: number
}

interface SimilarArticlesProps {
  articleId: number | string
  limit?: number
  compact?: boolean
}

// A search page renders one <SimilarArticles> per result and a chat source
// list renders one per source, so the same article id is fetched repeatedly
// within a single view -- and again on every subsequent search. The backend
// already caches these by `recommend:similar:{version}:{id}:{limit}`, but the
// browser still paid a full request each time. This module-level memo keys on
// the same (articleId, limit) pair so a repeated id is served from memory
// instead of the network. Successful non-empty responses only: an error or an
// empty result is not memoized, so a later mount can retry.
const similarCache = new Map<string, Article[]>()

function cacheKey(articleId: number | string, limit: number): string {
  return `${articleId}:${limit}`
}


export default function SimilarArticles({
  articleId,
  limit = 5,
  compact = false,
}: SimilarArticlesProps) {
  const [articles, setArticles] = useState<Article[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!articleId) return

    const key = cacheKey(articleId, limit)
    const memoized = similarCache.get(key)
    if (memoized) {
      setArticles(memoized)
      setLoading(false)
      setError(null)
      return
    }

    const controller = new AbortController()
    setLoading(true)
    setError(null)

    fetch(`${API_BASE}/recommend/similar/${articleId}?limit=${limit}`, {
      signal: controller.signal,
      headers: authHeaders(),
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load')
        return res.json()
      })
      .then((data) => {
        if (!controller.signal.aborted) {
          const next: Article[] = data.similar_articles || []
          if (next.length) similarCache.set(key, next)
          setArticles(next)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          setError(err.message)
          setLoading(false)
        }
      })

    return () => controller.abort()
  }, [articleId, limit])

  if (loading && articles.length === 0) {
    return compact ? (
      <span className={styles['similar-loading']}>Loading...</span>
    ) : null
  }

  if (error && articles.length === 0) {
    return compact ? null : null
  }

  if (!articles.length) {
    return null
  }

  const displayArticles = expanded ? articles : articles.slice(0, 3)

  if (compact) {
    return (
        <div className={styles['similar-articles-compact']}>
        <div className={styles['similar-heading']}>Similar articles</div>
        <div className={styles['similar-list']}>
          {displayArticles.map((article) => (
            <SafeArticleLink
              key={article.id}
              url={article.url}
              className={styles['similar-item']}
            >
              <span className={styles['similar-title']}>{article.title}</span>
              {article.category && (
                <span className={styles['similar-category']}>{article.category}</span>
              )}
            </SafeArticleLink>
          ))}
        </div>
        {articles.length > 3 && (
          <button
            type="button"
            className={styles['similar-show-more']}
            onClick={() => setExpanded((e) => !e)}
          >
            {expanded ? 'Show less' : `Show ${articles.length - 3} more`}
          </button>
        )}
      </div>
    )
  }

  return (
    <div className={styles['similar-articles']}>
      <div className={styles['similar-heading']}>Similar articles</div>
      <div className={styles['similar-list']}>
        {displayArticles.map((article) => (
          <SafeArticleLink key={article.id} url={article.url} className={styles['similar-card']}>
            <div className={styles['similar-card-content']}>
              <span className={styles['similar-card-title']}>{article.title}</span>
              {article.summary && (
                <p className={styles['similar-card-summary']}>{article.summary}</p>
              )}
              <div className={styles['similar-card-meta']}>
                {article.category && (
                  <span className={styles['similar-card-category']}>{article.category}</span>
                )}
                {article.published_date && (
                  <span className="similar-card-date">
                    {formatArticleDate(article.published_date)}
                  </span>
                )}
              </div>
            </div>
          </SafeArticleLink>
        ))}
      </div>
      {articles.length > 3 && (
        <button
          type="button"
          className={styles['similar-show-more']}
          onClick={() => setExpanded((e) => !e)}
        >
          {expanded ? 'Show less' : `Show ${articles.length - 3} more`}
        </button>
      )}
    </div>
  )
}
