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
 * rendered as an inert element, so a poisoned `url` (`javascript:`, `data:`,
 * `//evil/phish`) is never clickable — the same fallback `app/page.tsx` already
 * uses for search results.
 *
 * The fallback is a `div`, not a `span`: every caller passes block content
 * (`div`/`h2`/`p`), and a `span` may only contain phrasing content, so a `span`
 * wrapper would emit malformed server HTML. All three call sites style the
 * wrapper with an explicit `display`, so the element change is visually inert.
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
