import type { Metadata } from 'next'
import { headers } from 'next/headers'
import { Montserrat } from 'next/font/google'
import './globals.css'

// Per-request rendering is REQUIRED: the CSP nonce is generated fresh in
// middleware (crypto.randomUUID()) on every request and read here via
// headers(). Reading headers() already opts this layout into dynamic
// rendering, so this declaration is not what disables static builds — the
// nonce flow does. It cannot be scoped per-route because the nonce'd
// <script> must live in the root document <head> for CSP to be valid.
// This is an accepted tradeoff for a per-request CSP.
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