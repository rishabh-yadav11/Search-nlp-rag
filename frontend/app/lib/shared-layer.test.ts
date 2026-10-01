import { readFileSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

/**
 * Structural guard against a second copy of a shared helper creeping back: a
 * duplicate definition passes every behavioural test, so the invariant is
 * asserted against the source tree rather than at runtime — no import graph
 * reveals a second declaration.
 */

const FRONTEND_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')

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
  formatEpochDateTime: ['app/lib/format.ts'],
  formatClockTime: ['app/lib/format.ts'],
  formatCost: ['app/lib/format.ts'],
  logout: ['app/lib/auth.ts'],
}

/** Names replaced by a shared helper, with the module that replaced each: a page
 *  re-introducing one under its old name declares nothing the owner list sees. */
const RETIRED: Record<string, string> = {
  usd: 'app/lib/format.ts (formatCost)',
  sanitizeApiBase: 'app/lib/api-base.ts (sanitizeApiBaseOrigin)',
  safeNext: 'app/lib/safe-url.ts (isSafeRedirect)',
  relativeTime: 'app/lib/format.ts (formatEpochRelative)',
}

/**
 * Source files to scan: every TypeScript module under the frontend root, not
 * just `app/` — walking only `app/` silently moves every rule here out of
 * coverage the first time a top-level directory is added. Build output and
 * dependencies are skipped, as are test files: they legitimately name these
 * helpers and the env var in order to assert on them, so including them would
 * make the scan assert against itself.
 */
const SKIP_DIRS: Record<string, true> = {
  node_modules: true,
  '.next': true,
  out: true,
  coverage: true,
  dist: true,
}

function sourceFiles(): string[] {
  const found: string[] = []

  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (SKIP_DIRS[entry.name] === true || entry.name.startsWith('.')) continue
      const full = join(dir, entry.name)
      if (entry.isDirectory()) {
        walk(full)
      } else if (/\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)) {
        found.push(full)
      }
    }
  }

  walk(FRONTEND_ROOT)
  return found.map((f) => relative(FRONTEND_ROOT, f).split(sep).join('/'))
}

/** Strip comments but KEEP strings: a helper named in a comment is not a
 *  definition, while a page building a money string is exactly what the cost
 *  check must see. */
function codeOf(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/[^\n]*/g, '$1')
}

/**
 * True when `code` DECLARES `name`: a `function`/`class` declaration, a
 * `const`/`let`/`var` binding, or an object-literal member. Matching only
 * `function` and `const` was a hole — `let usd = …` and
 * `const helpers = { isSafeRedirect() {} }` both slipped past.
 *
 * A call or an import is NOT a declaration, which is what lets a page consume a
 * shared helper without tripping these checks: an imported name is followed by
 * `}` or `,` in a list, never by `:` or `(`.
 *
 * `allowObjectLiteral` must be false for `.tsx` files. In TSX `= {` is not
 * reliably an object literal: `<Card value={formatCost(x)} />` is textually
 * identical to `const x = {formatCost() {}}`, and JSX expression containers are
 * indistinguishable from object literals too. A plain `.ts` module has no JSX.
 */
function declares(code: string, name: string, allowObjectLiteral: boolean): boolean {
  const alternatives = [
    // function name( / async function name( / export default function name(
    `(?:^|[\\n{;])\\s*(?:export\\s+)?(?:default\\s+)?(?:async\\s+)?function\\s+${name}\\b`,
    // class name
    `(?:^|[\\n{;])\\s*(?:export\\s+)?(?:default\\s+)?(?:abstract\\s+)?class\\s+${name}\\b`,
    // const/let/var name
    `(?:^|[\\n{;])\\s*(?:export\\s+)?(?:const|let|var)\\s+${name}\\b`,
  ]
  if (allowObjectLiteral) {
    // Object-literal member: `= { name(` , `return { name: … }`.
    alternatives.push(`(?:=|,|\\(|return|:)\\s*\\{\\s*(?:async\\s+)?${name}\\s*[:(]`)
  }
  return new RegExp(alternatives.join('|')).test(code)
}

const files = sourceFiles()

describe('shared layer — exactly one definition of each helper', () => {
  it('scans every module under the frontend root', () => {
    // A silently-empty or wrongly-scoped scan makes every assertion vacuous.
    expect(files).toContain('middleware.ts')
    expect(files).toContain('app/page.tsx')
    expect(files).toContain('app/lib/auth.ts')
    // Root-level config and setup files: the walk must not stop at `app/`.
    expect(files).toContain('next.config.ts')
    expect(files).toContain('vitest.config.ts')
    expect(files).toContain('vitest.setup.ts')
    // Exclusions: test files and dependency/build directories.
    expect(files.some((f) => /\.test\.tsx?$/.test(f))).toBe(false)
    expect(files.some((f) => f.includes('node_modules') || f.startsWith('.'))).toBe(false)
    expect(files.length).toBeGreaterThan(10)
  })

  for (const [helper, owners] of Object.entries(OWNERS)) {
    it(`declares ${helper} in exactly one module`, () => {
      const declaredIn = files.filter((file) =>
        declares(codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8')), helper, !file.endsWith('.tsx')),
      )
      expect(declaredIn).toEqual(owners)
    })
  }

  for (const retired of Object.keys(RETIRED)) {
    it(`never re-declares the retired ${retired}()`, () => {
      // A helper that comes back under a different name declares nothing that
      // looks like one, so the owner list above cannot see it.
      const revived = files.filter((file) =>
        declares(codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8')), retired, !file.endsWith('.tsx')),
      )
      expect(revived, `${retired}() was replaced by ${RETIRED[retired]}`).toEqual([])
    })
  }
})

describe('shared layer — no second API-base resolver', () => {

  it('reads NEXT_PUBLIC_API_BASE in only one module', () => {
    // A second literal read is invisible to any behavioural test. Matched
    // against code with comments stripped, so merely MENTIONING the variable in
    // prose does not count as a reader.
    const readers = files.filter((file) =>
      /process\.env\.NEXT_PUBLIC_API_BASE/.test(codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))),
    )
    expect(readers).toEqual(['app/lib/api-base.ts'])
  })

  it('spells the dev loopback origin in only one module', () => {
    // Two copies of the dev origin is how the backend port changes in one place only.
    const spellings = files.filter((file) => {
      const code = codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))
      return code.includes('http://localhost:8001')
    })
    expect(spellings).toEqual(['app/lib/api-base.ts'])
  })
})

describe('shared layer — pages consume the shared helpers', () => {
  // Consolidation is not enough if the call sites were left behind.
  const read = (file: string): string => codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8'))

  it('no page renders a date or timestamp with its own locale call', () => {
    // Three shapes, because one regex cannot separate dates from number
    // grouping. The first shape must not require a locale argument: the exact
    // regression was `new Date(x).toLocaleDateString()` with NO argument, so a
    // rule keyed on `[]`/`''` locale missed it.
    const DATE_RENDERING: RegExp[] = [
      // `toLocaleDateString`/`toLocaleTimeString` are date-only callees whatever they are passed.
      /\.toLocale(?:Date|Time)String\(/,
      // A locale call handed an options bag naming a date field — the date options
      // are what make it a date (`maximumFractionDigits` is number formatting).
      /\.toLocale\w*String\([^)]*\{[^)]*\b(?:dateStyle|timeStyle|timeZone|weekday|era|year|month|day|hour|minute|second)\s*:/,
      // A Date expression feeding any locale call.
      /new Date\([^)]*\)[\s\S]{0,120}?\.toLocale\w*String\(/,
      // A bare `[]` / `''` locale, which takes the viewer's locale.
      /\.toLocale\w*String\(\s*(?:\[\]|''|"")/,
    ]
    // `Number(n).toLocaleString()` groups a number; it is out of scope for this rule.
    const offenders = files.filter(
      (file) => !OWNERS.formatDate.includes(file) && DATE_RENDERING.some((re) => re.test(read(file))),
    )
    expect(offenders).toEqual([])
  })

  it('no page formats a cost with its own currency string building', () => {
    // The two shapes the deleted copies used (`$${cost.toFixed(2)}` and
    // `'$' + Number(v).toLocaleString(…)`), matched precisely and scoped to
    // cost-shaped building rather than to every `$` in the tree: `app/chat/DataViz.tsx`
    // legitimately formats a CHART AXIS as `$${trim(v)}B` / `$M` / `₹ Cr`, which is a
    // unit abbreviation for backend chart data, not a per-session cost.
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

  it('wires the shared logout() to every log-out button', () => {
    // Matching the substring `logout` is worthless: these files match it through
    // a CSS class (`topbar-logout`) and through the import alone, so it would stay
    // green with the handler re-pointed at something local.
    //
    // The rule is "every log-out button is wired to `logout()`", not "only the
    // top bar has one" — the scan covers the whole frontend root, so a page that
    // grows its own log-out button later is still caught.
    const filesWithLogoutButton = files.filter((file) => /Log out|Log Out/.test(read(file)))
    expect(filesWithLogoutButton).toContain('app/components/TopBar.tsx')
    for (const file of filesWithLogoutButton) {
      expect(read(file), `${file} must wire its log-out button to logout()`).toMatch(
        /onClick=\{logout\}/,
      )
    }
  })

  it('the sign-out request is issued in only one module', () => {
    const posters = files.filter((file) => read(file).includes('/api/auth/logout'))
    expect(posters).toEqual(['app/lib/auth.ts'])
  })
})
