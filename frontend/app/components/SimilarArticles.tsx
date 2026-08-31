'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { API_BASE, authHeaders } from '../lib/auth'

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
          setArticles(data.similar_articles || [])
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
      <span className="similar-loading">Loading...</span>
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
      <div className="similar-articles-compact">
        <div className="similar-heading">Similar articles</div>
        <div className="similar-list">
          {displayArticles.map((article) => (
            <Link
              key={article.id}
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="similar-item"
            >
              <span className="similar-title">{article.title}</span>
              {article.category && (
                <span className="similar-category">{article.category}</span>
              )}
            </Link>
          ))}
        </div>
        {articles.length > 3 && (
          <button
            type="button"
            className="similar-show-more"
            onClick={() => setExpanded((e) => !e)}
          >
            {expanded ? 'Show less' : `Show ${articles.length - 3} more`}
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="similar-articles">
      <div className="similar-heading">Similar articles</div>
      <div className="similar-list">
        {displayArticles.map((article) => (
          <Link
            key={article.id}
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="similar-card"
          >
            <div className="similar-card-content">
              <span className="similar-card-title">{article.title}</span>
              {article.summary && (
                <p className="similar-card-summary">{article.summary}</p>
              )}
              <div className="similar-card-meta">
                {article.category && (
                  <span className="similar-card-category">{article.category}</span>
                )}
                {article.published_date && (
                  <span className="similar-card-date">
                    {new Date(article.published_date).toLocaleDateString()}
                  </span>
                )}
              </div>
            </div>
          </Link>
        ))}
      </div>
      {articles.length > 3 && (
        <button
          type="button"
          className="similar-show-more"
          onClick={() => setExpanded((e) => !e)}
        >
          {expanded ? 'Show less' : `Show ${articles.length - 3} more`}
        </button>
      )}
    </div>
  )
}
