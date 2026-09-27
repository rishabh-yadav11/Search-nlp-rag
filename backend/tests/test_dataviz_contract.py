"""The ONE dataviz validator contract, checked across both sides of the wire (#267).

The server and the browser each decide whether a ``dataviz`` block is valid, and
the two verdicts must be the same: the server strips a block it calls malformed
and bills a nudge retry for it, while the browser renders whatever it accepted.
When they disagreed, a streamed chart vanished on reload, and a value-less or
half-numeric block was billed a retry that could never help.

So this test runs ONE fixture corpus (fixtures/dataviz_corpus.json) through
BOTH implementations — the backend's ``parse_dataviz`` in-process, and the
frontend's real ``parseDataViz`` executed under node by importing
frontend/app/chat/datavizContract.ts — and fails if they accept different
fixtures, pick a different value column, or coerce a cell to a different number.
The three lists and grammars that used to be copy-pasted between the two files
(fence pattern, missing-value tokens, numeric-literal pattern) are asserted
equal as strings/sets too, so a divergence cannot hide in a corner the corpus
does not reach.

node is required (>= 22.6, for its built-in TypeScript support); without it the
cross-language half skips loudly rather than passing quietly, and the Python
half still runs.
"""

import json
import math
import pathlib
import shutil
import subprocess

import pytest

from app import chat as chat_module

_HERE = pathlib.Path(__file__).resolve().parent
CORPUS_PATH = _HERE / "fixtures" / "dataviz_corpus.json"
HARNESS_PATH = _HERE / "dataviz_harness.mjs"
CONTRACT_TS_PATH = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "app" / "chat" / "datavizContract.ts"


def _plotted_values(data):
    """The numbers a chart would plot, in row order: every non-missing cell of
    the value column, coerced. Mirrors the harness so the two sides are compared
    on the values they would actually draw, not merely on accept/reject."""
    vc = data.get("value_column")
    if vc is None:
        return []
    return [chat_module._as_float(row[vc]) for row in data["rows"] if not chat_module._missing_cell(row[vc])]


def _same_numbers(left, right):
    """Equality that treats NaN as equal to itself: a genuine NaN in the plotted
    values is a bug the corpus's expected verdict catches, not a disagreement
    between two implementations that both produced one."""
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


def _python_verdict(text):
    data = chat_module.parse_dataviz(text)
    if data is None:
        return {"accept": False, "value_column": None, "values": []}
    return {"accept": True, "value_column": data["value_column"], "values": _plotted_values(data)}


@pytest.fixture(scope="module")
def corpus():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frontend(corpus):
    """The shipped frontend validator's verdicts, by actually running it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH; cannot execute the frontend dataviz validator")
    proc = subprocess.run(
        [node, str(HARNESS_PATH), str(CONTRACT_TS_PATH), str(CORPUS_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"the frontend dataviz validator did not run (node {node}, exit {proc.returncode}); "
        f"node >= 22.6 is needed to import a .ts module\n{proc.stderr}"
    )
    return json.loads(proc.stdout)


def test_frontend_and_backend_agree_on_every_fixture(corpus, frontend):
    """The acceptance item: one corpus, both implementations, no fixture where
    they disagree. Each side's verdict, chosen value column and coerced plot
    values must match the other's."""
    disagreements = []
    for fixture in corpus["fixtures"]:
        py = _python_verdict(fixture["text"])
        ts = frontend["results"][fixture["name"]]
        if py["accept"] != ts["accept"] or not _same_numbers(py["values"], ts["values"]):
            disagreements.append(
                f"{fixture['name']}: accept backend={py['accept']} frontend={ts['accept']}, "
                f"values backend={py['values']} frontend={ts['values']} ({fixture['why']})"
            )
        elif py["accept"] and py["value_column"] != ts["value_column"]:
            disagreements.append(
                f"{fixture['name']}: value_column backend={py['value_column']} "
                f"frontend={ts['value_column']} ({fixture['why']})"
            )
    assert not disagreements, "the two validators disagree on:\n" + "\n".join(disagreements)


def test_every_fixture_has_the_verdict_both_sides_agree_on(corpus, frontend):
    """Agreement alone is not enough: if both sides changed the same wrong way,
    the comparison above would still pass. The corpus pins the CORRECT verdict,
    which is why a permissive 'accept everything' fix cannot slip through."""
    wrong = []
    for fixture in corpus["fixtures"]:
        for side, accept in (
            ("backend", _python_verdict(fixture["text"])["accept"]),
            ("frontend", frontend["results"][fixture["name"]]["accept"]),
        ):
            if ("accept" if accept else "reject") != fixture["expect"]:
                wrong.append(
                    f"{fixture['name']} ({side}): expected {fixture['expect']}, "
                    f"got {'accept' if accept else 'reject'} — {fixture['why']}"
                )
    assert not wrong, "verdicts the corpus says are wrong:\n" + "\n".join(wrong)


def test_both_sides_match_the_same_fence_text(corpus, frontend):
    """Same regex, same match extent, same captured payload — for every fixture.

    Comparing only the accept/reject verdict cannot see a whitespace class the
    two engines read differently: the class after the closing fence changes how
    much text a side swallows, not whether the block is valid, so the two could
    drift apart on every fixture while all the verdicts still agreed."""
    mismatched = []
    for fixture in corpus["fixtures"]:
        result = frontend["results"][fixture["name"]]
        match = chat_module._DATAVIZ_FENCE_RE.search(fixture["text"])
        expected = ([list(match.span()), match.group(1)] if match else [None, None])
        if [result["span"], result["captured"]] != expected:
            mismatched.append(
                f"{fixture['name']}: backend {expected} frontend {[result['span'], result['captured']]}"
            )
    assert not mismatched, "the two sides match the fence differently:\n" + "\n".join(mismatched)


def test_fixture_names_are_unique(corpus):
    """The harness keys its results by fixture name, so two fixtures sharing one
    would leave the first shadowed -- its verdict compared against the second's --
    and a real disagreement on it would never be looked at."""
    seen, duplicates = set(), []
    for fixture in corpus["fixtures"]:
        if fixture["name"] in seen:
            duplicates.append(fixture["name"])
        seen.add(fixture["name"])
    assert not duplicates, "fixture names used twice: " + ", ".join(sorted(duplicates))


def test_fence_grammar_is_one_string_on_both_sides(frontend):
    """The fence pattern is a single string, copied verbatim rather than
    re-derived, so it cannot drift while still meaning the same thing."""
    assert frontend["fence_src"] == chat_module.DATAVIZ_FENCE_PATTERN


def test_numeric_literal_grammar_is_one_string_on_both_sides(frontend):
    """A cell is a stated number only when the WHOLE cell is a plain numeric
    literal. The pattern is shared as a string so the frontend's Number() and
    the backend's float() cannot drift into accepting different spellings."""
    assert frontend["numeric_literal_src"] == chat_module._NUMERIC_LITERAL_SRC


def test_missing_value_tokens_are_identical_on_both_sides(corpus, frontend):
    """The 'value not stated' token list existed in two files with nothing
    keeping them equal; the corpus lists it once and both sides are compared
    against it, so adding a token to one side turns this red."""
    assert sorted(frontend["missing_value_tokens"]) == sorted(chat_module._MISSING_VALUE_TOKENS)
    assert sorted(corpus["missing_value_tokens"]) == sorted(chat_module._MISSING_VALUE_TOKENS)


def test_corpus_exercises_every_missing_value_token(corpus):
    """Each token gets a fixture whose ROWS ACTUALLY CONTAIN IT, and both sides
    read that fixture. A fixture merely *named* after a token would let a token
    be declared in all three lists while nothing tested it, which is the silent
    drift this whole corpus exists to prevent; so the token is looked for in the
    parsed cells, and dropping the token from either list flips the verdict of
    its own fixture."""
    by_name = {f["name"]: f["text"] for f in corpus["fixtures"]}
    problems = []
    for tok in chat_module._MISSING_VALUE_TOKENS:
        name = f"token_missing:{tok if tok else '<empty>'}"
        text = by_name.get(name)
        if text is None:
            problems.append(f"{tok!r}: no fixture named {name}")
            continue
        match = chat_module._DATAVIZ_FENCE_RE.search(text)
        cells = []
        if match is not None:
            try:
                payload = json.loads(match.group(1))
                cells = [cell for row in payload["rows"] for cell in row]
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                cells = []
        if tok not in cells:
            problems.append(f"{tok!r}: fixture {name} does not contain it as a cell (cells={cells!r})")
    assert not problems, "missing-value tokens no fixture really covers:\n" + "\n".join(problems)


def test_both_sides_trim_the_same_whitespace(frontend):
    """Which characters get trimmed off a cell, checked one codepoint at a time.

    The two languages disagree here and a string comparison cannot see it:
    JavaScript's trim() removes U+FEFF and Python's str.strip() does not, so a
    cell carrying a BOM read as "missing" in the browser and as a real value on
    the server. Both sides now name ASCII whitespace explicitly, and this pins
    every character either side might have an opinion about -- including the
    ones that must NOT be trimmed."""
    mismatched = [
        f"U+{cp.upper()}: backend trims={backend_trims} frontend trims={frontend['trim_probes'].get(cp)}"
        for cp, backend_trims in sorted(chat_module._TRIM_PROBES.items())
        if frontend["trim_probes"].get(cp) != backend_trims
    ]
    assert not mismatched, "the two sides trim different characters:\n" + "\n".join(mismatched)


def test_trim_grammar_is_one_string_on_both_sides(frontend):
    """The trim set is shared as a string, so the character class behind the
    per-codepoint probe above can only be changed in both places at once."""
    assert frontend["trim_src"] == chat_module._TRIM_SRC
