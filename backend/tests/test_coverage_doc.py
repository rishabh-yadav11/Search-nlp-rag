"""Check TEST_COVERAGE_GAPS.md against itself, in about a second.

The document used to quote one headline percentage that nothing re-measured, so
it was free to rot (issue #295). `scripts/measure_coverage.py` re-measures and
fails on drift, but it costs a full instrumented suite run -- two minutes -- and
it only tells you the document is wrong, not which number.

These tests are the cheap half. They import nothing from `app`, run nothing,
read no environment, and check the document's own claims against each other and
against the real source files on disk: a percentage has to follow from that
row's statements and missed count, the missed count has to equal the line
numbers the row lists, the totals have to equal the sum of the table, the
"uncovered error handling" bullets have to equal the `except`/`raise` lines
among the uncovered lines, and every hand-written gap entry has to repeat its
table row exactly and cite only lines that are actually uncovered.

A hand-edited number breaks exactly one of those, and the failure message names
the line and the expected value. `measure_coverage.py` remains the authority
for whether the numbers match a real run.

The document and `app/` are resolved from this file, never from the working
directory, so the result does not depend on where pytest was started.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
DOCUMENT = BACKEND / "TEST_COVERAGE_GAPS.md"
APP = BACKEND / "app"

# The block format has exactly one definition, in the gate: the writer, the
# gate's own comparison and these tests all read the same constants and the same
# line-compression helper, so a format change cannot be half-applied.
_spec = importlib.util.spec_from_file_location(
    "measure_coverage", BACKEND / "scripts" / "measure_coverage.py"
)
gate = importlib.util.module_from_spec(_spec)
# dataclasses resolves annotations through sys.modules while the class body is
# being executed, so the module has to be registered before it runs.
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

COVERAGE_BEGIN = gate.COVERAGE_BEGIN
COVERAGE_END = gate.COVERAGE_END
GAPS_BEGIN = gate.GAPS_BEGIN
GAPS_END = gate.GAPS_END

ERROR_HEADING = "### Uncovered error-handling statements"
NO_ERROR_STATEMENT = "No uncovered error-handling statements."
ERROR_COUNT_RE = re.compile(
    r"^`except` and `raise` statements that no test executes — "
    r"(?P<count>\d+) of the (?P<missed>\d+) uncovered statements:$"
)
ERROR_BULLET_RE = re.compile(r"^- `(?P<module>app/[^`]+\.py)`: (?P<lines>[\d, ]+)$")
OVERALL_RE = re.compile(
    r"^Overall: (?P<percent>\d+\.\d)% \((?P<missed>\d+) of (?P<statements>\d+) statements uncovered\)$"
)
SUITE_RE = re.compile(
    r"^Suite: \d+ passed, \d+ skipped, \d+ failed, \d+ errors, \d+ xfailed, \d+ xpassed$"
)
# The entry prefix without the prose tail, so "no prose at all" is reported as
# the missing prose it is rather than as an unparseable line. The figures are
# followed by `.` or `:` -- a period introduces a sentence of prose, a colon
# introduces the list of cited lines, and which one is used carries no claim the
# machine has to check.
GAP_PREFIX_RE = re.compile(
    r"^- `(?P<module>app/[^`]+\.py)` — (?P<percent>\d+\.\d)% "
    r"\((?P<missed>\d+)/(?P<statements>\d+) statements uncovered\)[.:](?P<rest>.*)$"
)
CONTINUATION_RE = re.compile(r"^  \S")
TABLE_ROW_RE = gate.ROW_RE
CITATION_RE = gate.LINE_CITATION_RE
MODULE_REF_RE = re.compile(r"\bapp/[\w./]+\.py\b")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
# Values that differ per machine or per run. The block has to be identical on
# every checkout, so none of these may appear anywhere in it.
ENVIRONMENT_VALUE_RES = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b[0-9a-f]{40}\b"),
    re.compile(r"\bPython \d"),
    re.compile(r"\b(pytest|coverage|ruff) \d"),
)


@dataclass(frozen=True)
class Section:
    """A marker-delimited block, with the document line each line came from."""

    lines: tuple[str, ...]
    first_line: int

    def number(self, index: int) -> int:
        return self.first_line + index

    def find(self, needle: str) -> int:
        for index, line in enumerate(self.lines):
            if line == needle:
                return index
        raise AssertionError(f"the generated block has no {needle!r} line")


@dataclass(frozen=True)
class TableRow:
    module: str
    percent: float
    statements: int
    missed: int
    lines_text: str
    uncovered: frozenset[int]
    line_number: int


@dataclass(frozen=True)
class GapEntry:
    module: str
    percent: float
    missed: int
    statements: int
    prose: str
    line_number: int
    text: str


def _read_document() -> str:
    assert DOCUMENT.is_file(), f"{DOCUMENT} does not exist"
    return DOCUMENT.read_text(encoding="utf-8")


def _section(text: str, begin: str, end: str) -> Section:
    lines = text.splitlines()
    for marker in (begin, end):
        count = lines.count(marker)
        assert count == 1, f"{DOCUMENT.name}: expected exactly one {marker} line, found {count}"
    start, stop = lines.index(begin), lines.index(end)
    assert start < stop, f"{DOCUMENT.name} line {start + 1}: {end} comes before {begin}"
    return Section(tuple(lines[start + 1 : stop]), start + 2)


def _coverage_section() -> Section:
    return _section(_read_document(), COVERAGE_BEGIN, COVERAGE_END)


def _gaps_section() -> Section:
    return _section(_read_document(), GAPS_BEGIN, GAPS_END)


def _table_rows(section: Section) -> list[TableRow]:
    rows: list[TableRow] = []
    for index, line in enumerate(section.lines):
        if not line.startswith("| `app/"):
            continue
        match = TABLE_ROW_RE.match(line)
        assert match is not None, (
            f"{DOCUMENT.name} line {section.number(index)}: {line!r} is not a table row; the "
            f"generated form is {gate.ROW_RE.pattern!r}"
        )
        lines_text = match["lines"]
        rows.append(
            TableRow(
                module=match["module"],
                percent=float(match["percent"]),
                statements=int(match["statements"]),
                missed=int(match["missed"]),
                lines_text=lines_text,
                uncovered=frozenset(gate.expand_ranges(lines_text)),
                line_number=section.number(index),
            )
        )
    return rows


def _app_modules() -> list[str]:
    return sorted(
        path.relative_to(BACKEND).as_posix()
        for path in APP.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _source(module: str) -> list[str]:
    return (BACKEND / module).read_text(encoding="utf-8").splitlines()


def _cited_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for group in CITATION_RE.findall(text):
        for item in group.split(","):
            bounds = [int(part) for part in item.strip().split("-")]
            ranges.append((bounds[0], bounds[-1]))
    return ranges


def _entry_lines(section: Section) -> list[tuple[int, str]]:
    """The gaps body without its leading HTML notice, as (document line, text)."""
    kept: list[tuple[int, str]] = []
    inside_comment = False
    for index, line in enumerate(section.lines):
        if inside_comment:
            inside_comment = "-->" not in line
            continue
        if line.startswith("<!--"):
            inside_comment = "-->" not in line
            continue
        if line.strip():
            kept.append((section.number(index), line))
    return kept


def _gap_entries(section: Section) -> tuple[list[GapEntry], list[str]]:
    """Parse the gaps body; one message per line that breaks the grammar."""
    entries: list[GapEntry] = []
    errors: list[str] = []
    open_index: int | None = None
    for number, line in _entry_lines(section):
        if line.startswith("-"):
            match = GAP_PREFIX_RE.match(line)
            if match is None:
                errors.append(
                    f"{DOCUMENT.name} line {number}: {line!r} is not a gap entry; expected "
                    "- `app/<path>.py` — <pct>% (<missed>/<statements> statements uncovered)."
                )
                open_index = None
                continue
            open_index = len(entries)
            entries.append(
                GapEntry(
                    module=match["module"],
                    percent=float(match["percent"]),
                    missed=int(match["missed"]),
                    statements=int(match["statements"]),
                    prose=match["rest"].strip(),
                    line_number=number,
                    text=line,
                )
            )
        elif CONTINUATION_RE.match(line):
            if open_index is None:
                errors.append(
                    f"{DOCUMENT.name} line {number}: {line!r} is a continuation line with no "
                    "entry above it; continuation lines belong to a `- ` entry"
                )
                continue
            entry = entries[open_index]
            entries[open_index] = GapEntry(
                module=entry.module,
                percent=entry.percent,
                missed=entry.missed,
                statements=entry.statements,
                # A long entry's prose wraps onto its continuation lines, so what
                # counts as the prose is everything after the figures.
                prose=f"{entry.prose} {line.strip()}".strip(),
                line_number=entry.line_number,
                text=f"{entry.text} {line.strip()}",
            )
        else:
            errors.append(
                f"{DOCUMENT.name} line {number}: {line!r} is neither a `- ` gap entry nor a "
                "continuation indented by exactly two spaces"
            )
            open_index = None
    return entries, errors


# --------------------------------------------------------------------------


def test_document_declares_both_machine_verified_blocks():
    """The two marker pairs exist, once each, in the order the document needs."""
    lines = _read_document().splitlines()
    positions = {}
    for marker in (COVERAGE_BEGIN, COVERAGE_END, GAPS_BEGIN, GAPS_END):
        count = lines.count(marker)
        assert count == 1, f"{DOCUMENT.name}: expected exactly one {marker} line, found {count}"
        positions[marker] = lines.index(marker) + 1
    ordered = [
        positions[COVERAGE_BEGIN],
        positions[COVERAGE_END],
        positions[GAPS_BEGIN],
        positions[GAPS_END],
    ]
    assert ordered == sorted(ordered), (
        f"{DOCUMENT.name}: markers out of order at lines {ordered}; the generated table has to "
        "come before the hand-written gaps it is checked against"
    )


def test_generated_block_is_machine_written():
    """The block opens with the gate's own notice, command, interpreter and suite line."""
    section = _coverage_section()
    assert len(section.lines) >= 4, (
        f"{DOCUMENT.name} line {section.first_line}: the generated block is empty or truncated "
        f"({len(section.lines)} lines); regenerate it with "
        "`python scripts/measure_coverage.py --write`"
    )
    for index, expected in enumerate((gate.GENERATED_NOTICE, gate.COMMAND_LINE)):
        assert section.lines[index] == expected, (
            f"{DOCUMENT.name} line {section.number(index)}: expected {expected!r}, "
            f"found {section.lines[index]!r}"
        )
    assert gate.MEASURED_WITH_RE.match(section.lines[2]), (
        f"{DOCUMENT.name} line {section.number(2)}: expected a {gate.MEASURED_WITH_PREFIX!r} "
        f"line naming the measuring interpreter, found {section.lines[2]!r}"
    )
    suite = section.lines[3]
    assert SUITE_RE.match(suite), (
        f"{DOCUMENT.name} line {section.number(3)}: expected "
        "'Suite: <n> passed, <n> skipped, <n> failed, <n> errors, <n> xfailed, <n> xpassed', "
        f"found {suite!r}"
    )


def test_generated_block_carries_no_environment_specific_values():
    """No date or commit SHA, and no version outside the one `Measured with:` line.

    The block must be identical on every checkout so a diff means the document
    rotted. That is true of everything here except the interpreter, and the
    interpreter was the one thing that had to be recorded: statement counts are
    parser-dependent (the same source yields 5137 statements under 3.11 and
    5075 under 3.14), so a block written on 3.14 and compared on CI's 3.11 was a
    62-statement diff whose cause appeared nowhere. Recording it converts an
    unreadable diff into a named mismatch, and the gate now fails with that
    name instead of the arithmetic.
    """
    section = _coverage_section()
    measured = [
        (index, line)
        for index, line in enumerate(section.lines)
        if gate.MEASURED_WITH_RE.match(line)
    ]
    assert len(measured) == 1, (
        f"{DOCUMENT.name}: expected exactly one {gate.MEASURED_WITH_PREFIX!r} line in the "
        f"generated block, found {len(measured)}; regenerate it with "
        "`python scripts/measure_coverage.py --write`"
    )
    exempt = {measured[0][0]}
    for index, line in enumerate(section.lines):
        if index in exempt:
            continue
        for pattern in ENVIRONMENT_VALUE_RES:
            found = pattern.search(line)
            assert found is None, (
                f"{DOCUMENT.name} line {section.number(index)}: {found.group(0)!r} makes the "
                f"generated block environment-specific (matched {pattern.pattern!r}); the block "
                "has to be byte-identical on every checkout, so it carries no clock, SHA or "
                "version outside its single `Measured with:` line"
            )


def test_every_app_module_has_exactly_one_table_row():
    """One row per `app/**.py`, in both directions, and no duplicates."""
    rows = _table_rows(_coverage_section())
    assert rows, f"{DOCUMENT.name}: the generated table has no rows for app/**/*.py"
    documented = [row.module for row in rows]
    duplicates = sorted(module for module, count in Counter(documented).items() if count > 1)
    assert not duplicates, (
        f"{DOCUMENT.name}: {len(duplicates)} module(s) have more than one table row: "
        f"{', '.join(duplicates)}; each app/**/*.py module gets exactly one row"
    )
    expected = _app_modules()
    missing = sorted(set(expected) - set(documented))
    extra = sorted(set(documented) - set(expected))
    assert not missing and not extra, (
        f"{DOCUMENT.name}: the table and the source tree disagree -- missing from the table: "
        f"{missing or 'none'}; in the table but not in app/: {extra or 'none'}"
    )


def test_table_rows_are_self_consistent():
    """Percent follows from the row, and the line column counts to `Missed`."""
    for row in _table_rows(_coverage_section()):
        expected_percent = gate.format_percent(row.statements - row.missed, row.statements)
        assert row.percent == expected_percent, (
            f"{DOCUMENT.name} line {row.line_number}: {row.module} says {row.percent:.1f}% but "
            f"{row.statements} statements with {row.missed} missed is {expected_percent:.1f}%"
        )
        if row.statements == 0:
            assert row.lines_text == "none", (
                f"{DOCUMENT.name} line {row.line_number}: {row.module} has no statements, so the "
                f"uncovered-lines column must read `none`, found {row.lines_text!r}"
            )
            continue
        assert row.missed > 0 or row.lines_text == "none", (
            f"{DOCUMENT.name} line {row.line_number}: {row.module} reports 0 missed but lists "
            f"{row.lines_text!r}"
        )
        assert len(row.uncovered) == row.missed, (
            f"{DOCUMENT.name} line {row.line_number}: {row.module} reports {row.missed} missed "
            f"statements but its uncovered-lines column {row.lines_text!r} lists "
            f"{len(row.uncovered)}"
        )
        limit = len(_source(row.module))
        beyond = sorted(line for line in row.uncovered if not 1 <= line <= limit)
        assert not beyond, (
            f"{DOCUMENT.name} line {row.line_number}: {row.module} has {limit} lines, so the "
            f"uncovered-lines column must not cite {beyond}"
        )


def test_uncovered_line_ranges_are_canonically_compressed():
    """The line column is exactly what `--write` produces, so a hand-edit shows up."""
    for row in _table_rows(_coverage_section()):
        canonical = gate.compress_ranges(row.uncovered)
        assert row.lines_text == canonical, (
            f"{DOCUMENT.name} line {row.line_number}: {row.module} lists its uncovered lines as "
            f"{row.lines_text!r}; the generated form is {canonical!r} (ascending, compressed to "
            "ranges, `none` when empty)"
        )


def test_overall_line_agrees_with_the_table():
    """`Overall:` is the sum of the rows, not a remembered number."""
    section = _coverage_section()
    rows = _table_rows(section)
    statements = sum(row.statements for row in rows)
    missed = sum(row.missed for row in rows)
    claimed = [
        (index, line) for index, line in enumerate(section.lines) if line.startswith("Overall:")
    ]
    assert len(claimed) == 1, (
        f"{DOCUMENT.name}: expected exactly one 'Overall:' line in the generated block, found "
        f"{len(claimed)}"
    )
    index, line = claimed[0]
    match = OVERALL_RE.match(line)
    assert match is not None, (
        f"{DOCUMENT.name} line {section.number(index)}: {line!r}; expected "
        "'Overall: <pct>% (<missed> of <statements> statements uncovered)'"
    )
    expected_percent = gate.format_percent(statements - missed, statements)
    assert float(match["percent"]) == expected_percent, (
        f"{DOCUMENT.name} line {section.number(index)}: says {match['percent']}% but the table "
        f"sums to {statements} statements with {missed} missed = {expected_percent:.1f}%"
    )
    assert (int(match["missed"]), int(match["statements"])) == (missed, statements), (
        f"{DOCUMENT.name} line {section.number(index)}: says {match['missed']} of "
        f"{match['statements']} statements uncovered, the table sums to {missed} of {statements}"
    )


def test_error_handling_section_lists_exactly_the_uncovered_error_statements():
    """The bullets are the `except`/`raise` lines among the table's uncovered lines."""
    section = _coverage_section()
    heading = section.find(ERROR_HEADING)
    rows = {row.module: row for row in _table_rows(section)}
    expected = {
        module: gate.error_handling_lines(_source(module), sorted(row.uncovered))
        for module, row in rows.items()
    }
    expected = {module: hits for module, hits in expected.items() if hits}
    total = sum(len(hits) for hits in expected.values())
    missed_total = sum(row.missed for row in rows.values())
    body = section.lines[heading + 1 :]

    if not expected:
        content = [line for line in body if line.strip()]
        assert content == [NO_ERROR_STATEMENT], (
            f"{DOCUMENT.name} line {section.number(heading + 1)}: with no uncovered `except` or "
            f"`raise` the section must hold the single line {NO_ERROR_STATEMENT!r}, "
            f"found {content!r}"
        )
        return

    # The count sentence is the first non-blank line under the heading.
    index = next(
        (offset for offset in range(heading + 1, len(section.lines)) if section.lines[offset].strip()),
        len(section.lines),
    )
    count_match = ERROR_COUNT_RE.match(section.lines[index])
    assert count_match is not None, (
        f"{DOCUMENT.name} line {section.number(index)}: expected "
        f"'`except` and `raise` statements that no test executes — {total} of the {missed_total} "
        f"uncovered statements:', found {section.lines[index:index + 1]!r}"
    )
    assert (int(count_match["count"]), int(count_match["missed"])) == (total, missed_total), (
        f"{DOCUMENT.name} line {section.number(index)}: says {count_match['count']} of "
        f"{count_match['missed']}, but the table's uncovered lines contain {total} `except` or "
        f"`raise` lines out of {missed_total} missed"
    )

    bullets = {
        match["module"]: [int(part) for part in match["lines"].split(",")]
        for match in (ERROR_BULLET_RE.match(line) for line in body)
        if match is not None
    }
    assert bullets == expected, (
        f"{DOCUMENT.name}: the uncovered error-handling bullets disagree with the table -- "
        f"expected {expected}, found {bullets}"
    )


def test_gap_entries_follow_the_documented_grammar():
    """`- ` module — pct% (missed/statements statements uncovered). prose, 2-space wraps."""
    section = _gaps_section()
    entries, errors = _gap_entries(section)
    assert not errors, "\n".join(errors)
    assert entries, (
        f"{DOCUMENT.name} line {section.first_line}: the gaps block lists no entries; every "
        "module with uncovered statements needs one"
    )
    for entry in entries:
        assert entry.prose, (
            f"{DOCUMENT.name} line {entry.line_number}: {entry.module!r} has no prose after its "
            "figures; an entry has to say what is missing, not repeat the table"
        )
    repeated = sorted(module for module, count in Counter(e.module for e in entries).items() if count > 1)
    assert not repeated, (
        f"{DOCUMENT.name}: {len(repeated)} module(s) have more than one gap entry: "
        f"{', '.join(repeated)}; one entry per module"
    )


def test_gap_modules_are_exactly_the_modules_with_uncovered_statements():
    """Both directions: no uncovered module may be omitted, no covered one listed."""
    rows = {row.module: row for row in _table_rows(_coverage_section())}
    entries, errors = _gap_entries(_gaps_section())
    assert not errors, "\n".join(errors)
    listed = {entry.module for entry in entries}
    with_gaps = {module for module, row in rows.items() if row.missed > 0}
    missing = sorted(with_gaps - listed)
    extra = sorted(listed - with_gaps)
    assert not missing and not extra, (
        f"{DOCUMENT.name}: the gaps block must list exactly the table rows with Missed > 0 -- "
        f"omitted: {missing or 'none'}; listed but fully covered: {extra or 'none'}"
    )


def test_gap_figures_match_the_table_row():
    """Each entry repeats its row's percent, missed and statements exactly."""
    rows = {row.module: row for row in _table_rows(_coverage_section())}
    entries, errors = _gap_entries(_gaps_section())
    assert not errors, "\n".join(errors)
    for entry in entries:
        row = rows.get(entry.module)
        assert row is not None, (
            f"{DOCUMENT.name} line {entry.line_number}: {entry.module!r} has no row in the "
            f"generated table; the table lists {', '.join(sorted(rows))}"
        )
        assert (entry.percent, entry.missed, entry.statements) == (
            row.percent,
            row.missed,
            row.statements,
        ), (
            f"{DOCUMENT.name} line {entry.line_number}: {entry.module} claims "
            f"{entry.percent:.1f}% ({entry.missed}/{entry.statements} statements uncovered); the "
            f"table row on line {row.line_number} says {row.percent:.1f}% "
            f"({row.missed}/{row.statements})"
        )


def test_gap_line_citations_point_at_uncovered_lines():
    """`lines A` / `lines A-B` must be inside the module and actually uncovered."""
    section = _gaps_section()
    entries, errors = _gap_entries(section)
    assert not errors, "\n".join(errors)
    rows = {row.module: row for row in _table_rows(_coverage_section())}
    for entry in entries:
        assert entry.module in rows, (
            f"{DOCUMENT.name} line {entry.line_number}: {entry.module} has no row in the "
            f"generated table, so its citations cannot be checked; the table lists "
            f"{', '.join(sorted(rows)) or 'nothing'}"
        )
        row = rows[entry.module]
        limit = len(_source(entry.module))
        for start, stop in _cited_ranges(entry.text):
            assert stop <= limit, (
                f"{DOCUMENT.name} line {entry.line_number}: {entry.module} is {limit} lines long, "
                f"so the citation `lines {start}-{stop}` is out of range"
            )
            assert any(start <= number <= stop for number in row.uncovered), (
                f"{DOCUMENT.name} line {entry.line_number}: {entry.module} cites lines "
                f"{start}-{stop}, none of which are uncovered; the table's uncovered lines are "
                f"{gate.compress_ranges(row.uncovered)}"
            )


def test_no_module_percentage_claim_outside_the_generated_blocks():
    """Only the generated block may pair an `app/**.py` module with a percentage."""
    lines = _read_document().splitlines()
    inside: set[int] = set()
    for begin, end in ((COVERAGE_BEGIN, COVERAGE_END), (GAPS_BEGIN, GAPS_END)):
        for index in range(lines.index(begin), lines.index(end) + 1):
            inside.add(index)
    offenders = [
        (index + 1, line)
        for index, line in enumerate(lines)
        if index not in inside and MODULE_REF_RE.search(line) and PERCENT_RE.search(line)
    ]
    assert not offenders, (
        f"{DOCUMENT.name}: a coverage percentage paired with a module name outside the "
        "machine-verified blocks is the rot this document already suffered -- move the claim "
        "into the generated table or the gaps block, or drop the number:\n"
        + "\n".join(f"  line {number}: {line}" for number, line in offenders)
    )


def _counts(passed: int, failed: int = 0, errors: int = 0) -> dict:
    return {name: 0 for name in gate.SUITE_COUNTS} | {
        "passed": passed,
        "failed": failed,
        "errors": errors,
    }


def test_a_stale_document_can_actually_be_regenerated():
    """`--write` must not be blocked by the very staleness it exists to repair.

    The block is generated and `tests/test_coverage_doc.py` polices it, so a
    stale document fails those tests -- and they are part of the run that
    produces the measurement. Treating that as a broken measurement deadlocks
    the gate: `--write` exits 2, the document is never touched, and the only
    way out is moving the test file aside by hand.
    """
    output = (
        "FAILED tests/test_coverage_doc.py::test_overall_line_agrees_with_the_table\n"
        "FAILED tests/test_coverage_doc.py::test_gap_figures_match_the_table_row\n"
        "2 failed, 1899 passed in 120.00s"
    )
    tolerated = gate.staleness_failures(output, _counts(1899, failed=2))
    assert tolerated == (
        "tests/test_coverage_doc.py::test_overall_line_agrees_with_the_table",
        "tests/test_coverage_doc.py::test_gap_figures_match_the_table_row",
    )


@pytest.mark.parametrize(
    ("summary", "counts", "why"),
    [
        (
            "FAILED tests/test_auth.py::test_login\n1 failed, 1900 passed in 120.00s",
            _counts(1900, failed=1),
            "a failure anywhere else is a real failure, not staleness",
        ),
        (
            "ERROR tests/test_coverage_doc.py\n1 error, 1900 passed in 120.00s",
            _counts(1900, errors=1),
            "a collection error is not a stale number",
        ),
        (
            "FAILED tests/test_coverage_doc.py::test_a\n1 failed, 1900 passed in 120.00s",
            _counts(1900, failed=2),
            "the summary must account for every count pytest reported",
        ),
        (
            "Interrupted: 1 error during collection\n1 error",
            _counts(0, errors=1),
            "output with no usable short summary cannot be vouched for",
        ),
    ],
)
def test_a_broken_measurement_is_never_tolerated(summary, counts, why):
    """Fail closed: `--write` must leave the document alone for anything else.

    A gate that writes the block from a run it could not vouch for is worse
    than no gate, so every other reason a run can be red has to be refused.
    """
    assert gate.staleness_failures(summary, counts) is None, why


def test_gate_refuses_to_run_off_the_target_interpreter(monkeypatch, capsys):
    """A block measured elsewhere does not describe this source.

    Statement counts are interpreter-dependent -- 3.11 and 3.14 disagree on
    byte-identical source -- so a `--write` from the wrong interpreter produces
    a document that passes locally and is rejected by CI. That is how this gate
    stayed red on every push to main, so the refusal is checked here rather
    than left to a comment describing an intent the code does not enforce.
    """
    monkeypatch.setattr(gate.sys, "version_info", (3, 14, 7, "final", 0))
    assert gate.main([]) == 3
    assert gate.main(["--write"]) == 3
    errors = capsys.readouterr().err
    assert "python3.11" in errors
    assert "3.14.7" in errors


def test_measured_with_comparison_ignores_the_coverage_version():
    """Only the interpreter decides a match; the tool version is provenance.

    The coverage.py version is recorded so a reader can tell what produced the
    block, but it was measured to make no difference to statement counts, so
    comparing it would report a mismatch that is not one.
    """
    matched = gate.MEASURED_WITH_RE.match("Measured with: Python 3.11.16, coverage 7.16.2")
    assert matched is not None
    assert matched.group("interp") == "Python 3.11.16"
    assert matched.group("interp") == gate.MEASURED_WITH_RE.match(
        "Measured with: Python 3.11.16, coverage 7.15.4"
    ).group("interp")
