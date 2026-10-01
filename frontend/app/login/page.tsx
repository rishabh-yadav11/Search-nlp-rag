'use client'

import { Suspense, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { API_BASE, authRequestInit, clearLegacyToken, clearMeCache, getMe } from '../lib/auth'
import { isSafeRedirect } from '../lib/safe-url'

function LoginForm() {
  const router = useRouter()
  const params = useSearchParams()
  // Honor `next` only when it is a safe, same-origin, root-relative path; an unvalidated value would let an attacker redirect the victim off-site after login.
  const rawNext = params.get('next') || '/chat'
  const next = isSafeRedirect(rawNext) ? rawNext : '/chat'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    // The session is an httpOnly cookie JS cannot read, so `/api/auth/me` is the only way to know; a network failure is not a logout and must not redirect.
    getMe()
      .then((me) => {
        if (me) router.replace(next)
      })
      .catch(() => {
      })
    return () => {
      mountedRef.current = false
      abortRef.current?.abort()
    }
  }, [router, next])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    const trimmed = email.trim()
    if (!trimmed || !password) {
      setError('Email and password are required.')
      return
    }
    setBusy(true)

    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort(), 15000)

    try {
      const res = await fetch(
        `${API_BASE}/api/auth/login`,
        authRequestInit({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: trimmed, password }),
          signal: controller.signal,
        })
      )
      clearTimeout(timeout)
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError((body as { detail?: string }).detail ?? `Login failed (${res.status}).`)
        return
      }
      // The session arrives as an httpOnly cookie, so there is nothing to read or persist in JS; drop the previous session's cached user and any legacy token.
      clearMeCache()
      clearLegacyToken()
      router.replace(next)
      return
    } catch (err) {
      clearTimeout(timeout)
      // The cleanup may have aborted this request; don't set state on an unmounted component.
      if (!mountedRef.current) return
      if (controller.signal.aborted) {
        setError('The request timed out. Please try again.')
      } else {
        setError('Could not reach the server. Please try again.')
      }
      return
    } finally {
      if (mountedRef.current) setBusy(false)
    }
  }

  return (
    <div className="auth-wrap">
      <form className="auth-card" onSubmit={submit}>
        <h1 className="auth-title">Sign in</h1>
        <p className="auth-sub">ASK VCCircle</p>
        <label className="auth-field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />
        </label>
        <label className="auth-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
            minLength={8}
          />
        </label>
        {error ? (
          <div className="auth-error" role="alert">
            {error}
          </div>
        ) : null}
        <button type="submit" className="auth-btn" disabled={busy}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
        <p className="auth-alt">
          No account? <Link href="/signup">Create one</Link>
        </p>
      </form>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  )
}