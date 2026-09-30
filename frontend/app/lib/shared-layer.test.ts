import { readFileSync, readdirSync } from 'node:fs'
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

/**
 * Names this issue deleted, with the module that replaced each. A page that
 * re-introduces one of these under its old name is the exact regression the
 * issue reported, and the two shape-based checks below cannot see it: an
 * `usd()` that returns `'$' + v.toFixed(2)` declares no `formatCost` and
 * builds no pattern they match.
 */
const RETIRED: Record<string, string> = {
  usd: 'app/lib/format.ts (formatCost)',
  sanitizeApiBase: 'app/lib/api-base.ts (sanitizeApiBaseOrigin)',
  safeNext: 'app/lib/safe-url.ts (isSafeRedirect)',
  relativeTime: 'app/lib/format.ts (formatEpochRelative)',
}

/**
 * Source files to scan: every TypeScript module under the frontend root —
 * `app/`, `middleware.ts`, the configs, and any directory added later.
 *
 * The whole root, not just `app/`: walking only `app/` means the first person
 * to add a top-level `lib/` or `hooks/` directory silently moves every rule in
 * this file out of coverage, and a guard that quietly stops guarding is worse
 * than no guard. Build output and dependencies are skipped, as are test
 * files — they legitimately name these helpers and the env var in order to
 * assert on them, so including them would make the scan assert against itself.
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

/**
 * Strip comments, but KEEP string literals. The checks below match source
 * text: a helper named in a comment is not a definition, whereas a page
 * building a money string is exactly what the cost check must see, so
 * stripping strings would defeat it.
 */
function codeOf(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/[^\n]*/g, '$1')
}

/**
 * True when `code` DECLARES `name`, in any of the forms a duplicate could
 * take: a `function` or `class` declaration, a `const`/`let`/`var` binding, or
 * an object-literal property or method. Matching only `function` and `const`
 * was a hole — `let usd = …` and `const helpers = { isSafeRedirect() {} }`
 * both slipped past the owner and retired-name checks.
 *
 * A call or an import of the same name is NOT a declaration, which is what
 * lets a page consume a shared helper without tripping these checks: an
 * imported name is followed by `}` or `,` in a list, never by `:` or `(`.
 *
 * `allowObjectLiteral` must be false for `.tsx` files. In TSX, `= {` is not
 * reliably an object literal: `<Card value={formatCost(x)} />` is textually
 * identical to `const x = {formatCost() {}}`, and JSX expression containers
 * (`{formatCost(x)}`) are indistinguishable from object literals too. In a
 * plain `.ts` module there is no JSX, so the same pattern is safe there.
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

/** Every frontend module, scanned once. */
const files = sourceFiles()

describe('shared layer — exactly one definition of each helper', () => {
  it('scans every module under the frontend root', () => {
    // A silently-empty or wrongly-scoped scan would make every assertion in
    // this file vacuously true, so pin both the breadth and the exclusions.
    expect(files).toContain('middleware.ts')
    expect(files).toContain('app/page.tsx')
    expect(files).toContain('app/lib/auth.ts')
    // Root-level config and setup files, not just `app/`: the walk must not
    // stop at the first directory.
    expect(files).toContain('next.config.ts')
    expect(files).toContain('vitest.config.ts')
    expect(files).toContain('vitest.setup.ts')
    // Exclusions: test files (they name these helpers on purpose) and
    // dependency/build directories.
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
      // The issue's own symptom was a helper that came back under a different
      // name, which the owner list above cannot see: `usd` is not `formatCost`
      // and declares nothing that looks like one.
      const revived = files.filter((file) =>
        declares(codeOf(readFileSync(join(FRONTEND_ROOT, file), 'utf8')), retired, !file.endsWith('.tsx')),
      )
      expect(revived, `${retired}() was replaced by ${RETIRED[retired]}`).toEqual([])
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

  it('no page renders a date or timestamp with its own locale call', () => {
    // Three shapes, because one regex cannot separate dates from number
    // grouping — and the FIRST shape must not require a locale argument. The
    // exact regression issue #301 was filed about is
    // `new Date(article.published_date).toLocaleDateString()`, with NO
    // argument at all, so a rule keyed on `[]`/`''` locale missed it.
    const DATE_RENDERING: RegExp[] = [
      // `toLocaleDateString` / `toLocaleTimeString` are date-only callees by
      // definition, whatever they are passed.
      /\.toLocale(?:Date|Time)String\(/,
      // A locale call handed an options bag naming a date field. The date
      // options are what make this a date: `maximumFractionDigits` in
      // `app/chat/DataViz.tsx` is number formatting and must not match.
      /\.toLocale\w*String\([^)]*\{[^)]*\b(?:dateStyle|timeStyle|timeZone|weekday|era|year|month|day|hour|minute|second)\s*:/,
      // A Date expression feeding any locale call, e.g. the dashboard's
      // `new Date(ts * 1000).toLocaleString([], { dateStyle: … })`.
      /new Date\([^)]*\)[\s\S]{0,120}?\.toLocale\w*String\(/,
      // A bare `[]` / `''` locale, which takes the viewer's locale.
      /\.toLocale\w*String\(\s*(?:\[\]|''|"")/,
    ]
    // `Number(n).toLocaleString()` and `tokens.toLocaleString()` group a
    // number; they are not date rendering and are not in scope for this rule.
    const offenders = files.filter(
      (file) => !OWNERS.formatDate.includes(file) && DATE_RENDERING.some((re) => re.test(read(file))),
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

  it('wires the shared logout() to every log-out button', () => {
    // Asserting the substring `logout` is worthless on its own: every one of
    // these files matches it through a CSS class (`topbar-logout`) and through
    // the import alone, so it would stay green with the handler re-pointed at
    // something local. Require the button to actually be wired to it.
    //
    // The log-out control lives in the shared top bar, which every app route
    // mounts, so the bar is the file that must hold it: if that button ever
    // disappears, every route loses its sign-out at once. The scan covers the
    // whole frontend root, so a page that grows its own log-out button later
    // is still caught by the loop below — the rule is "every button is wired",
    // not "only the bar has one".
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
