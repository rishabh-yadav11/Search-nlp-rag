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
 * Renders a backend-supplied `url` as a link only when `isSafeUrl` approves both
 * scheme and origin; otherwise the children render inert, so a poisoned `url`
 * (`javascript:`, `data:`, `//evil/phish`) is never clickable. The fallback is a
 * `div`, not a `span`: callers pass block content, and a `span` may only contain
 * phrasing content.
 */
export default function SafeArticleLink({ url, className, children }: SafeArticleLinkProps) {
  if (!isSafeUrl(url)) {
    return <div className={className}>{children}</div>
  }
  return (
    <Link href={url} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
    </Link>
  )
}
