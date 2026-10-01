'use client'

import { Suspense, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { API_BASE, getMe } from '../lib/auth'
import { isSafeRedirect } from '../lib/safe-url'

function SignupForm() {
  const router = useRouter()
  const params = useSearchParams()
  // Honor `next` only when it is a safe, same-origin, root-relative path; anything the shared guard refuses falls back to /chat.
  const rawNext = params.get('next')
  const next = isSafeRedirect(rawNext) ? rawNext : '/chat'
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [success, setSuccess] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    // The session is an httpOnly cookie JS cannot read, so `/api/auth/me` is the only way to know; a network failure is not a logout and must not redirect.
    getMe()
      .then((me) => {
        if (me) router.replace(next)
      })
      .catch(() => {
      })
  }, [router, next])

  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  function validate(): string {
    if (!email.trim() || !password) return 'Email and password are required.'
    if (password.length < 8) return 'Password must be at least 8 characters.'
    if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) {
      return 'Password must contain a letter and a digit.'
    }
    return ''
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (busy) return
    setError('')
    setSuccess('')
    const problem = validate()
    if (problem) {
      setError(problem)
      return
    }
    setBusy(true)
    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort(), 15000)
    try {
      const res = await fetch(`${API_BASE}/api/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password, name: name.trim() }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError((body as { detail?: string }).detail ?? `Sign up failed (${res.status}).`)
        return
      }
      // The server answers 200 identically for a fresh and an already-registered address, so this form cannot be used to probe which emails are registered.
      const data = (await res.json()) as { message?: string }
      setSuccess(
        data.message ||
          'If this email is not already registered, your account is ready. Sign in with your email and password to continue.',
      )
    } catch (err) {
      if (controller.signal.aborted) {
        setError('Request timed out. Please try again.')
      } else {
        setError('Could not reach the server. Please try again.')
      }
    } finally {
      clearTimeout(timeout)
      setBusy(false)
    }
  }

  if (success) {
    const signInHref = `/login?next=${encodeURIComponent(next)}`
    return (
      <div className="auth-wrap">
        <div className="auth-card">
          <h1 className="auth-title">Almost there</h1>
          <p className="auth-sub">{success}</p>
          <button type="button" className="auth-btn" onClick={() => router.push(signInHref)}>
            Sign in
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-wrap">
      <form className="auth-card" onSubmit={submit}>
        <h1 className="auth-title">Create account</h1>
        <p className="auth-sub">Join ASK VCCircle</p>
        <label className="auth-field">
          <span>Name (optional)</span>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
        </label>
        <label className="auth-field">
          <span>Email</span>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" required />
        </label>
        <label className="auth-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
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
          {busy ? 'Creating account…' : 'Sign up'}
        </button>
        <p className="auth-alt">
          Already have an account? <Link href="/login">Sign in</Link>
        </p>
      </form>
    </div>
  )
}

export default function SignupPage() {
  return (
    <Suspense fallback={null}>
      <SignupForm />
    </Suspense>
  )
}