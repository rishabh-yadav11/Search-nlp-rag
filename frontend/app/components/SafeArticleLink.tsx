'use client'

import Link from 'next/link'
import type { ReactNode } from 'react'
import { isSafeUrl } from '../lib/safe-url'

interface SafeArticleLinkProps {
  /** Backend-supplied article URL. Never trusted. */
  url: string
  className?: string
  children: ReactNode
}

/**
 * Renders a backend-supplied article `url` as a link only when `isSafeUrl`
 * approves both the scheme and the origin. Otherwise the same children are
 * rendered as an inert `span`, so a poisoned `url` (`javascript:`, `data:`,
 * `//evil/phish`) is never clickable — the same fallback `app/page.tsx` already
 * uses for search results.
 */
export default function SafeArticleLink({ url, className, children }: SafeArticleLinkProps) {
  if (!isSafeUrl(url)) {
    return <span className={className}>{children}</span>
  }
  return (
    <Link href={url} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
    </Link>
  )
}
