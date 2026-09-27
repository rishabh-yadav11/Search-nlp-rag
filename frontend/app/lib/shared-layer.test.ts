import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

/**
 * The enforcement half of the shared API/format layer (issue #301).
 *
 * Consolidating a helper is only worth anything if nothing quietly grows a
 * second copy, and the copies are easy to grow by accident: a page needs a date
 * label, writes `new Date(x).toLocaleDateString()`, and the drift is back. The
 * behavioural tests cannot catch that — a second copy of a correct function
 * passes every one of them. So this file asserts the structural invariant
 * directly against the source tree.
 *
 * It is a source scan rather than a runtime probe on purpose: the failure this
 * guards against is a duplicate DEFINITION, which no import graph reveals.
 */

const FRONTEND_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const APP_DIR = join(FRONTEND_ROOT, 'app')

/** The modules that are allowed to own each helper. */
const OWNERS: Record<string, string[]> = {
  isSafeRedirect: ['app/lib/safe-url.ts'],
  sanitizeApiBaseOrigin: ['app/lib/api-base.ts'],
  parseApiBaseUrl: ['app/lib/api-base.ts'],
  readApiBaseEnv: ['app/lib/api-base.ts'],
  devApiBase: ['app/lib/api-base.ts'],
  parseLocalDate: ['app/lib/format.ts'],
  formatDate: ['app/lib/format.ts'],
  formatArticleDate: ['app/lib/format.ts'],
  formatEpochRelative: ['app/lib/format.ts'],
  formatCost: ['app/lib/format.ts'],
  logout: ['app/lib/auth.ts'],
}

/**
 * Source files to scan: every app module plus the edge middleware. Test files
 * are excluded — they legitimately name these helpers and the env var in order
 * to assert on them, so including them would make the scan assert against
 * itself.
 */
function sourceFiles(): string[] {
  const found: string[] = [join(FRONTEND_ROOT, 'middleware.ts')]

  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) {
        walk(full)
      } else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
        found.push(full)
      }
    }
  }

  walk(APP_DIR)
  return found.map((f) => relative(FRONTEND_ROOT, f).split(sep).join('/'))
}

/**
 * Strip comments, but KEEP string literals. The checks below match source
 * text: a helper named in a comment is not a definition, whereas a page
 * building a money string is exactly what the cost check must see, so
 * stripping strings would defeat it.
 */
function codeOf(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/[^\n]*/g, '$1')
}

/** Every frontend module, scanned once. */
const files = sourceFiles()

describe('shared layer — exactly one definition of each helper', () => {
  it('scans every frontend module, middleware included', () => {
    // A silently-empty scan would make every assertion below vacuously true.
    expect(files).toContain('middleware.ts')
    expect(files).toContain('app/page.tsx')
    expect(files).toContain('app/lib/auth.ts')
    expect(files.length).toBeGreaterThan(10)
  })

  for (const [helper, owners] of Object.entries(OWNERS)) {
    it(`declares ${helper} in exactly one module`, () => {
      const declaredIn = files.filter((file) => {
        const code = codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))
        // A declaration, not a call or an import: `function helper`, or
        // `const helper =`.
        return new RegExp(
          `(?:^|\\n)\\s*(?:export\\s+)?(?:async\\s+)?function\\s+${helper}\\b|(?:^|\\n)\\s*(?:export\\s+)?const\\s+${helper}\\b`,
        ).test(code)
      })

      expect(declaredIn).toEqual(owners)
    })
  }
})

describe('shared layer — no second API-base resolver', () => {

  it('reads NEXT_PUBLIC_API_BASE in only one module', () => {
    // A second literal read is the bug this whole module was created to fix,
    // and it is invisible to any behavioural test. Matched against code with
    // comments stripped, so a module that merely MENTIONS the variable in
    // prose is not counted as a reader.
    const readers = files.filter((file) =>
      /process\.env\.NEXT_PUBLIC_API_BASE/.test(codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))),
    )
    expect(readers).toEqual(['app/lib/api-base.ts'])
  })

  it('spells the dev loopback origin in only one module', () => {
    // Two copies of `http://localhost:8001` is how the backend port ends up
    // changed in one place only.
    const spellings = files.filter((file) => {
      const code = codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))
      return code.includes('http://localhost:8001')
    })
    expect(spellings).toEqual(['app/lib/api-base.ts'])
  })
})

describe('shared layer — pages consume the shared helpers', () => {
  // The other half of the acceptance criteria: consolidation is not enough if
  // the call sites were left behind, which is exactly the state the issue
  // reported (signup importing nothing, cards open-coding a date).
  const read = (file: string): string => codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))

  it('no page formats an article date with its own toLocaleDateString call', () => {
    const offenders = files.filter(
      (file) => !OWNERS.formatDate.includes(file) && /\.toLocaleDateString\(/.test(read(file)),
    )
    expect(offenders).toEqual([])
  })

  it('no page formats a cost with its own currency string building', () => {
    // The two shapes the deleted copies used, matched precisely:
    //   `$${cost.toFixed(2)}`      (app/chat/page.tsx)
    //   '$' + Number(v).toLocaleString(…)  (analytics/dashboard/page.tsx)
    // Scoped to cost-shaped building rather than to every `$` in the tree,
    // because `app/chat/DataViz.tsx` legitimately formats a CHART AXIS as
    // `$${trim(v)}B` / `$M` / `₹ Cr` — that is a unit abbreviation for backend
    // chart data, not a per-session cost, and it is deliberately untouched.
    const offenders = files.filter((file) => {
      if (OWNERS.formatCost.includes(file)) return false
      const code = read(file)
      return (
        /\$\$\{[^}]*\.toFixed/.test(code) ||
        /['"`]\$['"`]\s*\+\s*[^;\n]*toLocaleString/.test(code)
      )
    })
    expect(offenders).toEqual([])
  })

  it('every page with a log-out button calls the shared logout()', () => {
    const pagesWithLogoutButton = files.filter((file) => /Log out|Log Out/.test(read(file)))
    expect(pagesWithLogoutButton.length).toBeGreaterThan(1)
    for (const file of pagesWithLogoutButton) {
      expect(read(file)).toMatch(/logout/)
    }
  })

  it('the sign-out request is issued in only one module', () => {
    const posters = files.filter((file) => read(file).includes('/api/auth/logout'))
    expect(posters).toEqual(['app/lib/auth.ts'])
  })
})
