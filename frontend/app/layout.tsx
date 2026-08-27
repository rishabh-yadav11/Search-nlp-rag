import type { Metadata } from 'next'
import { headers } from 'next/headers'
import { Montserrat } from 'next/font/google'
import './globals.css'

export const dynamic = 'force-dynamic'

const montserrat = Montserrat({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700', '800'],
  variable: '--font-montserrat',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'ASK VCCircle',
  description: 'Hybrid retrieval · optional cited answers',
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const nonce = (await headers()).get('x-csp-nonce') ?? ''

  return (
    <html lang="en" className={montserrat.variable}>
      <head>
        <script nonce={nonce} dangerouslySetInnerHTML={{ __html: '' }} />
      </head>
      <body>{children}</body>
    </html>
  )
}