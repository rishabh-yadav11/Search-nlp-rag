'use client'

import { useEffect, useState } from 'react'
import SafeArticleLink from './SafeArticleLink'
import { formatArticleDate } from '../lib/format'
import { fetchSimilarArticles, peekSimilarArticles } from '../lib/similar'
import type { SimilarArticle } from '../lib/similar'
import { RequestTimeoutError } from '../lib/deadline'
import styles from './SimilarArticles.module.css'

interface SimilarArticlesProps {
  articleId: number | string
  limit?: number
  compact?: boolean
}


export default function SimilarArticles({
  articleId,
  limit = 5,
  compact = false,
}: SimilarArticlesProps) {
  const [articles, setArticles] = useState<SimilarArticle[]>(
    () => peekSimilarArticles(articleId, limit) ?? []
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  // Bumped by the error state's Retry button and part of the fetch effect's
  // deps, so a retry re-runs the request with a fresh deadline.
  const [retryCount, setRetryCount] = useState(0)

  useEffect(() => {
    if (!articleId) return

    // No AbortController here on purpose. The request is shared with every
    // other card in this view, so aborting it because THIS card unmounted
    // would take the rest of the view's data down with it. Unmounting only
    // means the answer is no longer worth applying. The deadline that bounds
    // the socket therefore lives in `app/lib/similar.ts`, on the shared
    // request itself: a backend that accepts the connection and never answers
    // would otherwise pin every card in the view on "Loading..." forever.
    let active = true
    setLoading(true)
    setError(null)

    fetchSimilarArticles(articleId, limit)
      .then((list) => {
        if (active) {
          setArticles(list)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (active) {
          setError(
            err instanceof RequestTimeoutError
              ? 'Similar articles did not load in time.'
              : err.message
          )
          setLoading(false)
        }
      })
      .finally(() => deadline.clear())

    return () => {
      active = false
    }
  }, [articleId, limit, retryCount])

  if (loading && articles.length === 0) {
    return compact ? (
      <span className={styles['similar-loading']}>Loading...</span>
    ) : null
  }

  if (error && articles.length === 0) {
    return (
      <div className={styles['similar-error']} role="alert">
        {error}
        <button
          type="button"
          className={styles['similar-retry']}
          onClick={() => setRetryCount((n) => n + 1)}
        >
          Retry
        </button>
      </div>
    )
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
