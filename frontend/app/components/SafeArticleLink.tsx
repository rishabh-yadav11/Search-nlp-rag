'use client'

import Link from 'next/link'
import type { ReactNode } from 'react'
import { isSafeUrl } from '../lib/safe-url'

interface SafeArticleLinkProps {
  
  url: string
  className?: string
  children: ReactNode
}


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
