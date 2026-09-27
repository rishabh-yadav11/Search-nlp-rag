// Runs the SHIPPED frontend dataviz validator over the shared corpus and
// prints its verdicts as JSON, so the backend's cross-language contract test
// (test_dataviz_contract.py) can compare them with parse_dataviz's own.
//
//   node [--experimental-strip-types] dataviz_harness.mjs <datavizContract.ts> [corpus.json]
//
// The module is IMPORTED, not copied or re-typed, so this measures the code the
// browser actually runs. It is a plain .mjs with a dynamic import of a .ts
// file, which node handles with its built-in TypeScript support — no bundler,
// no node_modules, no JSX (the validator has none by design).
//
// The caller probes the node version before getting here and passes
// --experimental-strip-types on the versions that need it to opt in (the flag
// is accepted as a no-op on newer ones, so it is always safe to pass).
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [modulePath, corpusPath] = process.argv.slice(2)
if (!modulePath || !corpusPath) {
  console.error('usage: node dataviz_harness.mjs <datavizContract.ts> <corpus.json>')
  process.exit(2)
}

const contract = await import(pathToFileURL(modulePath).href)
const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'))

// The coerced value of every non-missing cell in the value column: the exact
// numbers a chart would plot. Comparing the whole list, not just the first
// cell, is what catches the two sides reading the SAME cell differently
// (e.g. "1_000" as 1000 on one side and 1 on the other).
function plottedValues(block) {
  if (block.value_column == null) return []
  const out = []
  for (const row of block.rows) {
    const cell = row[block.value_column]
    if (!contract.isMissing(cell)) out.push(contract.toNum(cell))
  }
  return out
}

const results = {}
for (const fixture of corpus.fixtures) {
  const m = new RegExp(contract.FENCE_SRC).exec(fixture.text)
  const block = contract.parseDataViz(fixture.text)
  results[fixture.name] = {
    // The full match extent and the captured payload, not just the verdict: a
    // whitespace class that the two regex engines read differently changes how
    // much text a side swallows without changing whether the block is accepted,
    // so a verdict-only comparison cannot see it. The offsets are JavaScript
    // string indices, i.e. UTF-16 code units; the Python side converts to match.
    span: m ? [m.index, m.index + m[0].length] : null,
    captured: m ? m[1] : null,
    accept: !!block,
    value_column: block ? block.value_column : null,
    values: block ? plottedValues(block) : [],
  }
}

const tokens = contract.MISSING_VALUE_TOKENS
// Which characters the frontend trims off a cell. The backend spells its own
// trim set out in Python; comparing them one character at a time catches a
// divergence that an equal-looking string cannot (JS trim() eats U+FEFF, Python
// str.strip() does not).
const trimProbes = {}
for (const cp of [0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x200b, 0x2028,
  0x2029, 0x202f, 0x205f, 0x3000, 0xfeff]) {
  const ch = String.fromCodePoint(cp)
  trimProbes[cp.toString(16).padStart(4, '0')] = contract.trimCell(`x${ch}`) === 'x' && contract.trimCell(`${ch}x`) === 'x'
}

process.stdout.write(JSON.stringify({
  fence_src: contract.FENCE_SRC,
  numeric_literal_src: contract.NUMERIC_LITERAL_SRC,
  trim_src: contract.TRIM_SRC,
  max_json_depth: contract.MAX_JSON_DEPTH,
  trim_probes: trimProbes,
  missing_value_tokens: tokens instanceof Set ? [...tokens] : Object.keys(tokens),
  results,
}))
