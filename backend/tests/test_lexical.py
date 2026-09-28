"""The shared Jaccard helper.

`jaccard` existed as two independent copies (the /search read path and the chat
eval runner) plus a third ratio inlined in `_pairwise_source_jaccard`. The eval
runner's copy had lost the empty-set guard; it only avoided dividing by zero
because both of its call sites happened to pre-filter empty sets.

These tests pin the guard, the ratio, that every caller reaches one function
object, and that the AST contains exactly one such definition. The stopword
lists are intentionally NOT unified here -- see `app/lexical.py` -- so
`test_eval_runner_prompt_grouping_is_unchanged` stands in as the tripwire for
the one that feeds a reported metric.
"""

import ast
from pathlib import Path

import pytest

from app.lexical import jaccard
from scripts.eval_runner import _pairwise_source_jaccard


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (frozenset(), frozenset({"a"})),
        (frozenset({"a"}), frozenset()),
        (frozenset(), frozenset()),
    ],
)
def test_jaccard_empty_set_returns_zero(a, b):
    assert jaccard(a, b) == 0.0


def test_jaccard_overlap_ratio():
    assert jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == 0.5


def test_jaccard_is_one_function_across_callers():
    """Every caller must resolve the *same* function object.

    What this pins: the call sites in ``app/diversity.py`` and
    ``scripts/eval_runner.py`` all reach the guarded implementation, so a
    future edit cannot move one of them onto a different definition without
    failing here.

    On its own this is not enough, and it is worth recording why. Mutation
    testing showed this assertion alone stays green for two regressions: a
    module defining a *private* ``_jaccard`` the ``jaccard`` name never points
    at, and the ratio re-inlined in ``_pairwise_source_jaccard`` (that
    function's ``if a or b`` pair filter makes the both-empty case
    unreachable, so the inlined ratio and the guarded ``jaccard`` return the
    same number for every input that can actually reach the mean). Both are
    caught instead by the AST invariants
    ``test_exactly_one_jaccard_definition_and_it_keeps_the_guard`` and
    ``test_no_inlined_jaccard_ratio_outside_the_shared_helper``, which was
    verified by re-running both mutations and confirming they go red.
    """
    from app import diversity
    from scripts import eval_runner

    assert diversity.jaccard is jaccard
    assert eval_runner.jaccard is jaccard


def _result(*ids):
    return {"sources": [{"id": i} for i in ids]}


def test_pairwise_source_jaccard_matches_shared_jaccard():
    results = [_result(1, 2, 3), _result(2, 3, 4), _result(1, 5)]
    expected = (
        sum(
            [
                jaccard(frozenset({1, 2, 3}), frozenset({2, 3, 4})),
                jaccard(frozenset({1, 2, 3}), frozenset({1, 5})),
                jaccard(frozenset({2, 3, 4}), frozenset({1, 5})),
            ]
        )
        / 3
    )
    assert _pairwise_source_jaccard(results) == pytest.approx(expected)


def test_pairwise_source_jaccard_none_when_all_sources_empty():
    assert _pairwise_source_jaccard([_result(), _result(), _result()]) is None


def _backend_sources():
    """Every non-test backend module, as (relative path, parsed tree).

    Tests are excluded on purpose: they legitimately name these helpers to
    exercise them, and counting a test's own reference as a "second copy"
    would make the check meaningless.
    """
    root = Path(__file__).resolve().parent.parent
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] in {"tests", "venv"} or "__pycache__" in rel.parts:
            continue
        yield rel.as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def _function_defs(wanted):
    return [
        (name, node)
        for name, tree in _backend_sources()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]


def test_exactly_one_jaccard_definition_and_it_keeps_the_guard():
    """Structural invariant: one `jaccard` body, and it owns the empty-set guard.

    Behavioural tests cannot enforce this. Mutation testing showed that
    re-inlining the ratio in `_pairwise_source_jaccard`, or adding a private
    `_jaccard` that the module's `jaccard` name never points at, leaves every
    behavioural test green -- the first because the `if a or b` pair filter
    makes the both-empty case unreachable, the second because an unused copy
    is invisible at runtime. The divergence this issue is about is a
    *duplication*, so it has to be asserted on the syntax tree.
    """
    defs = _function_defs({"jaccard", "_jaccard"})
    assert [name for name, _ in defs] == ["app/lexical.py"], f"expected one jaccard definition, found {defs}"
    guards = [n for _, fn in defs for n in ast.walk(fn) if isinstance(n, ast.If)]
    assert guards, "the single jaccard definition lost its empty-set guard"


def _is_intersection(node):
    """True for a set intersection, bare or wrapped in ``len(...)``.

    The real source is ``len(a & b) / len(a | b)``, so the BitAnd sits inside
    a Call rather than directly under the division -- matching only the bare
    form would make this check vacuous.
    """
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.BitAnd)
    if isinstance(node, ast.Call):
        return any(_is_intersection(arg) for arg in node.args)
    return False


def _is_union_len(node):
    """True for ``len(a | b)`` -- the denominator that makes the ratio a Jaccard.

    The union denominator is load-bearing: ``topk_overlap`` in
    ``scripts/rerank_bench.py`` is ``len(sa & sb) / k``, an overlap-at-k that
    must NOT be folded into the shared ``jaccard`` -- its denominator is the
    cutoff, not the union. Requiring a union here keeps that call out.
    """
    if not isinstance(node, ast.Call):
        return False
    return any(isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.BitOr) for arg in node.args)


def test_no_inlined_jaccard_ratio_outside_the_shared_helper():
    """The `intersection / union` ratio must appear exactly once, in `jaccard`."""
    hits = [
        (name, node.lineno)
        for name, tree in _backend_sources()
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _is_intersection(node.left)
        and _is_union_len(node.right)
    ]
    outside = [(n, ln) for n, ln in hits if n != "app/lexical.py"]
    assert outside == [], f"a jaccard ratio was inlined outside app/lexical.py at {outside}"
    assert hits, "the shared jaccard ratio itself was not found -- the pattern is too narrow"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("climate change policy", "climate change so much"),
        ("covid vaccine rollout", "covid vaccine not yet"),
        ("bond yield curve", "bond yield no doubt"),
    ],
)
def test_eval_runner_prompt_grouping_is_unchanged(left, right):
    """These pairs must stay in separate groups.

    Tripwire for the eval runner's stopword list, which was deliberately left
    un-unified. Widening that list raises Jaccard (it rises monotonically as
    tokens are dropped), which merges these pairs and changes the reported
    cross-variation consistency number even though nothing about the answers
    or citations changed. Each of these pairs sits just under
    SIMILARITY_THRESHOLD and flips to at-or-above it once "so"/"not"/"no" join
    the stopword set, so asserting the grouping outcome -- not the token count
    -- is what actually catches the drift.
    """
    from scripts import eval_runner

    groups = eval_runner.group_prompts([eval_runner.normalize_prompt(p) for p in (left, right)])
    assert len(groups) == 2, f"{left!r} and {right!r} merged into one group: {groups}"
