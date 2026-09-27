// Runs the SHIPPED frontend dataviz validator over the shared corpus and
// prints its verdicts as JSON, so the backend's cross-language contract test
// (test_dataviz_contract.py) can compare them with parse_dataviz's own.
//
//   node dataviz_harness.mjs <datavizContract.ts> [corpus.json]
//
// The module is IMPORTED, not copied or re-typed, so this measures the code the
// browser actually runs. It is a plain .mjs with a dynamic import of a .ts
// file, which node handles with its built-in TypeScript support — no bundler,
// no node_modules, no JSX (the validator has none by design).
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
  const block = contract.parseDataViz(fixture.text)
  results[fixture.name] = block
    ? { accept: true, value_column: block.value_column, values: plottedValues(block) }
    : { accept: false, value_column: null, values: [] }
}

const tokens = contract.MISSING_VALUE_TOKENS
process.stdout.write(JSON.stringify({
  fence_src: contract.FENCE_SRC,
  numeric_literal_src: contract.NUMERIC_LITERAL_SRC,
  missing_value_tokens: tokens instanceof Set ? [...tokens] : Object.keys(tokens),
  results,
}))

