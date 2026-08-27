'use client'

import { Suspense, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { API_BASE, getToken, setToken } from '../lib/auth'

function SignupForm() {
  const router = useRouter()
  const params = useSearchParams()
  const next = params.get('next') || '/chat'
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (getToken()) router.replace(next)
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
      const data = (await res.json()) as { token: string }
      setToken(data.token)
      router.replace(next)
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