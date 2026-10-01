"""The ONE dataviz validator contract, checked across both sides of the wire.

The server and the browser each decide whether a ``dataviz`` block is valid, so
one fixture corpus runs through BOTH implementations -- the backend's
``parse_dataviz`` in-process, and the frontend's ``parseDataViz`` executed under
node -- and they must agree on the verdict, the value column, every coerced
cell, and the matched fence span.

Running the frontend half needs a node that can import a ``.ts`` file (built-in
TypeScript support, unflagged from 22.18). Because this module is the only
coverage of the browser-side validator, a node that cannot run it is a FAILURE
under CI and a skip locally, never a silent pass.
"""

import json
import math
import os
import pathlib
import re
import shutil
import subprocess

import pytest

from app import chat as chat_module

_HERE = pathlib.Path(__file__).resolve().parent
CORPUS_PATH = _HERE / "fixtures" / "dataviz_corpus.json"
HARNESS_PATH = _HERE / "dataviz_harness.mjs"
CONTRACT_TS_PATH = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "app" / "chat" / "datavizContract.ts"

# The oldest node whose TypeScript support can import the .ts validator.
MIN_NODE = (22, 6)


def _node_version(executable: str) -> tuple[int, ...] | None:
    """The node version as a comparable tuple, or None when it cannot be
    determined — including when the executable does not exist, which
    subprocess.run reports by raising rather than by a non-zero exit."""
    try:
        proc = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)", proc.stdout or "")
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _plotted_values(data):
    """Every non-missing cell of the value column, coerced, in row order --
    mirroring the harness so the sides are compared on the values they would
    draw, not merely on accept/reject."""
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


def _utf16_units(text: str, index: int) -> int:
    """``index`` (a Python codepoint offset) as a UTF-16 code-unit offset.

    JavaScript string indices count UTF-16 code units, so a character outside
    the BMP -- an emoji, say -- counts as TWO there and as ONE here; without this
    conversion any text carrying one fabricates a span divergence. surrogatepass
    so a stray surrogate in a fixture cannot make this helper raise."""
    return len(text[:index].encode("utf-16-le", "surrogatepass")) // 2


@pytest.fixture(scope="module")
def corpus():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frontend(corpus):
    """The shipped frontend validator's verdicts, by actually running it.

    A node too old to import a .ts module is treated exactly like a missing one:
    a failure under CI, a skip locally, so a developer on node 18 is not told
    their backend is broken while CI still keeps the only browser-side coverage."""
    node = shutil.which("node")
    unusable = None
    if node is None:
        unusable = f"node is not on PATH, or is older than {MIN_NODE[0]}.{MIN_NODE[1]}"
    else:
        version = _node_version(node)
        if version is None or version < MIN_NODE:
            usable = version and ".".join(str(p) for p in version) or "an unreadable version"
            unusable = f"node {usable} cannot import a .ts module; {MIN_NODE[0]}.{MIN_NODE[1]}+ is required"
    if unusable is not None:
        message = (
            f"{unusable}, so the shipped frontend dataviz validator cannot be executed and the "
            "cross-language half of #267 is untested. It needs node >= 22.6 for the built-in "
            "TypeScript support that imports frontend/app/chat/datavizContract.ts directly."
        )
        if os.environ.get("CI"):
            pytest.fail(message + " Install a new enough node in CI rather than letting the suite pass without it.")
        pytest.skip(message)
    # --experimental-strip-types opts 22.6-22.17 into type stripping; newer node
    # takes it as a no-op, so it is always safe to pass.
    proc = subprocess.run(
        [node, "--experimental-strip-types", str(HARNESS_PATH), str(CONTRACT_TS_PATH), str(CORPUS_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"the frontend dataviz validator did not run (node {node}, exit {proc.returncode})\n{proc.stderr}"
    )
    return json.loads(proc.stdout)


def test_node_probe_rejects_versions_that_cannot_import_typescript():
    """The gate is the version, not merely `which node`.

    A node 18 or 20 box passes a `shutil.which` check and then dies inside the
    harness on an unknown .ts extension — turning the whole backend suite red
    with a raw subprocess error instead of a clean, explanatory skip."""
    assert MIN_NODE == (22, 6)
    assert _node_version("definitely-not-a-real-node-binary") is None


def test_frontend_and_backend_agree_on_every_fixture(corpus, frontend):
    """One corpus, both implementations, no fixture where the verdict, the
    chosen value column or the coerced plot values differ."""
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


def test_fixture_names_are_unique(corpus):
    """The harness keys its results by fixture name, so a shared name leaves the
    first shadowed -- its verdict compared against the second's -- and a real
    disagreement on it would never be looked at."""
    seen, duplicates = set(), []
    for fixture in corpus["fixtures"]:
        if fixture["name"] in seen:
            duplicates.append(fixture["name"])
        seen.add(fixture["name"])
    assert not duplicates, "fixture names used twice: " + ", ".join(sorted(duplicates))


def test_both_sides_match_the_same_fence_text(corpus, frontend):
    """Same regex, same match extent, same captured payload — for every fixture.

    Comparing only the accept/reject verdict cannot see a whitespace class the
    two engines read differently: the class after the closing fence changes how
    much text a side swallows, not whether the block is valid, so the two could
    drift apart on every fixture while all the verdicts still agreed."""
    mismatched = []
    for fixture in corpus["fixtures"]:
        result = frontend["results"][fixture["name"]]
        text = fixture["text"]
        match = chat_module._DATAVIZ_FENCE_RE.search(text)
        if match is None:
            expected = [None, None]
        else:
            start, end = match.span()
            expected = [[_utf16_units(text, start), _utf16_units(text, end)], match.group(1)]
        if [result["span"], result["captured"]] != expected:
            mismatched.append(
                f"{fixture['name']}: backend {expected} frontend {[result['span'], result['captured']]}"
            )
    assert not mismatched, "the two sides match the fence differently:\n" + "\n".join(mismatched)


def test_deeply_nested_payload_is_dropped_not_raised():
    """A payload nested deep enough to exhaust the parser must be DROPPED, not
    raise.

    json.loads raises RecursionError on a deeply nested payload, and it used to
    escape parse_dataviz, then _sanitize_dataviz and _finalize_answer -- turning
    one malformed block into a failed chat request instead of a stripped one. The
    payload walk also caps the depth long before this, so the except clause is the
    second line of defence; it is pinned here because nothing else reaches it."""
    deep = "[" * 100_000 + "]" * 100_000
    text = '```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1, "note": ' + deep + "}\n```"
    assert chat_module.parse_dataviz(text) is None
    finalized = chat_module._finalize_answer(f"Prose.\n\n{text}", "show me a table of top deals")
    assert "```dataviz" not in finalized
    assert "Prose." in finalized


def test_view_pinning_applies_the_same_load_rules():
    """The view-pinning path re-loads the block itself, and that second load must
    apply the same rules as parse_dataviz: laxer, it returns a block unpinned and
    a user who asked for a bar chart quietly loses the chart."""
    payload = '{"columns": ["A", "B"], "rows": [["x", 1e999]], "value_column": 1}'
    assert chat_module._parse_dataviz_with_view(f"```dataviz\n{payload}\n```", "bar") is None
    bare = '{"columns": ["A", "B"], "rows": [["x", 1]], "value_column": 1, "note": Infinity}'
    assert chat_module._parse_dataviz_with_view(f"```dataviz\n{bare}\n```", "table") is None
    good = '{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}'
    pinned = chat_module._parse_dataviz_with_view(f"```dataviz\n{good}\n```", "bar")
    assert pinned is not None and pinned["view"] == "bar"


def test_fence_grammar_is_one_string_on_both_sides(frontend):
    """The fence pattern is a single string, copied verbatim rather than
    re-derived, so it cannot drift while still meaning the same thing."""
    assert frontend["fence_src"] == chat_module.DATAVIZ_FENCE_PATTERN


def test_numeric_literal_grammar_is_one_string_on_both_sides(frontend):
    """A cell is a stated number only when the WHOLE cell is a plain numeric
    literal; the pattern is shared as a string so the frontend's Number() and the
    backend's float() cannot drift into accepting different spellings."""
    assert frontend["numeric_literal_src"] == chat_module._NUMERIC_LITERAL_SRC


def test_trim_grammar_is_one_string_on_both_sides(frontend):
    """The trim set is shared as a string, so the character class behind the
    per-codepoint probe below can only be changed in both places at once."""
    assert frontend["trim_src"] == chat_module._TRIM_SRC


def test_payload_depth_limit_is_one_number_on_both_sides(frontend):
    """The nesting depth past which a payload counts as malformed is shared, so
    the frontend's recursive walk cannot give up where the backend's still
    succeeds (or overflow V8's stack on a payload json.loads refused)."""
    assert frontend["max_json_depth"] == chat_module._MAX_JSON_DEPTH


def test_both_sides_trim_the_same_whitespace(frontend):
    """Which characters get trimmed off a cell, checked one codepoint at a time.

    The two languages disagree here and a string comparison cannot see it:
    JavaScript's trim() removes U+FEFF and Python's str.strip() does not, so both
    sides name ASCII whitespace explicitly, and this pins every character either
    side might have an opinion about -- including the ones that must NOT be
    trimmed."""
    mismatched = [
        f"U+{cp.upper()}: backend trims={backend_trims} frontend trims={frontend['trim_probes'].get(cp)}"
        for cp, backend_trims in sorted(chat_module._TRIM_PROBES.items())
        if frontend["trim_probes"].get(cp) != backend_trims
    ]
    assert not mismatched, "the two sides trim different characters:\n" + "\n".join(mismatched)


def test_missing_value_tokens_are_identical_on_both_sides(corpus, frontend):
    """The 'value not stated' token list existed in two files with nothing
    keeping them equal; the corpus lists it once and both sides are compared
    against it, so adding a token to one side turns this red."""
    assert sorted(frontend["missing_value_tokens"]) == sorted(chat_module._MISSING_VALUE_TOKENS)
    assert sorted(corpus["missing_value_tokens"]) == sorted(chat_module._MISSING_VALUE_TOKENS)


def test_corpus_exercises_every_missing_value_token(corpus):
    """Each token gets a fixture whose ROWS ACTUALLY CONTAIN IT, and both sides
    read that fixture.

    A fixture merely *named* after a token would let a token be declared in all
    three lists while nothing tested it, which is the silent drift this whole
    corpus exists to prevent; so the token is looked for in the parsed cells, and
    dropping the token from either list flips the verdict of its own fixture."""
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
