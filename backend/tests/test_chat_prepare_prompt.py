"""Both chat prepare paths share ONE prompt-assembly tail.

``_prepare_turn`` (single entity) and ``_prepare_multi_entity_turn`` (comparison /
intersection) each used to end in their own copy of the same code: join the
fenced article blocks, render the user prompt, render the system prompt, wrap
the ``PreparedTurn``. That tail is now ``_prompt_turn``, leaving a single
``CHAT_PROMPT.format`` call site.

These tests pin three things the refactor must not change:

* the exact prompt bytes each path produces (golden files captured from the
  pre-refactor code),
* the retrieval-side behaviour deliberately LEFT per path, because it is
  genuinely different -- single-entity gates the merged list then slices
  ``[: k]``, multi-entity gates per entity with no such slice,
* the KeyError hazard a new ``CHAT_PROMPT`` field used to pose: with a format
  call in each path, updating only one copy made that path render and the other
  raise at REQUEST time, so the two paths could supply DIFFERENT field sets.
"""

import pathlib
import string
from typing import ClassVar

import pytest
from _support import run_sync as _run

from app import chat as chat_module
from app import main as main_module
from app.main import SourceArticle
from app.query_intent import MultiEntityQuery

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"

QUESTION = "Compare Alpha and Beta funding in 2024"
BODIES = ["Alpha body paragraph. " * 40, "Beta body paragraph. " * 40, "Gamma body paragraph. " * 40]
# id 2 scores 0.1: it clears ASK_MIN_SCORE_FACETED (0.0) but not ASK_MIN_SCORE
# (0.2), so a faceted turn keeps it and an unfaceted one drops it.
ARTICLES = [
    SourceArticle(id=1, title="Alpha funding", url="http://x/1", published_date="2024-03-01",
                  summary="Alpha summary", body=BODIES[0], score=0.91),
    SourceArticle(id=2, title="Mid funding", url="http://x/2", published_date="2024-05-02",
                  summary="Mid summary", body=BODIES[1], score=0.1),
    SourceArticle(id=3, title="Gamma funding", url="http://x/3", published_date="2024-07-03",
                  summary="Gamma summary", body=BODIES[2], score=0.64),
]
HISTORY = [
    chat_module.MessageOut(id=1, role="user", content="Earlier: which startups raised in 2024?",
                           sources=[], created_at=1000.0),
    chat_module.MessageOut(id=2, role="assistant", content="Alpha and Beta both raised in 2024.",
                           sources=[], created_at=1001.0),
]


def _golden(name):
    """The pre-refactor prompt for one turn, split into its two halves."""
    raw = (GOLDEN_DIR / name).read_text()
    answer, system = raw.split("=== SYSTEM (instruction turn) ===\n")
    return answer.removeprefix("=== ANSWER (user turn) ===\n").removesuffix("\n"), system.removesuffix("\n")


def _template_fields(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


class _RecordingTemplate(str):
    """A CHAT_PROMPT that remembers the fields of every ``format`` call on it."""

    calls: ClassVar[list[dict]] = []

    def format(self, *args, **kwargs):
        type(self).calls.append(dict(kwargs))
        return str.format(self, *args, **kwargs)


@pytest.fixture
def frozen(monkeypatch):
    """Pin every knob the two paths read, so prompts are reproducible."""
    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module, "_effective_chat_k", lambda q: 2)
    monkeypatch.setattr(chat_module, "detect_multi_entity", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", True)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(chat_module.config, "ASK_MIN_SCORE", 0.2)
    monkeypatch.setattr(chat_module.config, "ASK_MIN_SCORE_FACETED", 0.0)
    monkeypatch.setattr(chat_module.config, "CHAT_BODY_CHAR_LIMIT", 500)
    monkeypatch.setattr(chat_module.config, "CHAT_TOTAL_BODY_CHARS", 900)
    monkeypatch.setattr(chat_module.config, "CHAT_MAX_SOURCES", 6)
    monkeypatch.setattr(main_module, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    async def _rescue(q, articles):
        return articles

    monkeypatch.setattr(main_module, "body_rescue", _rescue)
    return monkeypatch


def _retrieval(monkeypatch, dealtype=None, industry=None, content_type=None, pool=None):
    pool = ARTICLES if pool is None else pool

    async def fake(retrieval_q, top_k, **kwargs):
        return (list(pool), dealtype, industry, content_type)

    monkeypatch.setattr(main_module, "retrieve_with_auto_facet_fallback", fake)


def _multi(mode="comparison", entities=("Alpha", "Beta")):
    return MultiEntityQuery(mode=mode, entities=list(entities), scaffold="funding")


def _single_turn(monkeypatch, dealtype=None):
    _retrieval(monkeypatch, dealtype=dealtype)
    return _run(chat_module._prepare_turn(QUESTION, list(HISTORY)))


def _multi_turn(monkeypatch, mode="comparison", dealtype=None):
    _retrieval(monkeypatch, dealtype=dealtype)
    return _run(chat_module._prepare_multi_entity_turn(_multi(mode), QUESTION, list(HISTORY)))


# --- the prompts themselves, byte for byte ---------------------------------


def test_single_entity_prompt_is_byte_identical_to_pre_refactor(frozen):
    """The single-entity turn must render exactly the prompt it rendered before
    the tail was extracted, not merely a prompt that formats without error."""
    turn = _single_turn(frozen)
    answer, system = _golden("chat_prompt_single_entity.txt")
    assert turn.answer == answer
    assert turn.system == system


def test_multi_entity_comparison_prompt_is_byte_identical_to_pre_refactor(frozen):
    turn = _multi_turn(frozen, "comparison")
    answer, system = _golden("chat_prompt_multi_entity_comparison.txt")
    assert turn.answer == answer
    assert turn.system == system


def test_multi_entity_intersection_prompt_is_byte_identical_to_pre_refactor(frozen):
    """The comparison/intersection instruction difference is intentional, so the
    two golden system prompts must differ from each other in exactly that half."""
    turn = _multi_turn(frozen, "intersection")
    answer, system = _golden("chat_prompt_multi_entity_intersection.txt")
    assert turn.answer == answer
    assert turn.system == system
    _, comparison = _golden("chat_prompt_multi_entity_comparison.txt")
    assert "## Multi-entity comparison" in comparison
    assert "## Multi-entity intersection" in system
    assert comparison != system


# --- the new-field KeyError hazard -----------------------------------------


def test_both_paths_supply_the_same_chat_prompt_fields(frozen, monkeypatch):
    """Both paths must hand CHAT_PROMPT the same complete set of fields.

    This is the hazard the shared tail closes. With a format call in each path,
    adding a field to the template and updating only one copy left that path
    rendering and the other raising KeyError at REQUEST time — two paths, two
    field sets, silently. A field is now supplied on both paths or on neither.
    """
    recorder = _RecordingTemplate(chat_module.CHAT_PROMPT)
    _RecordingTemplate.calls = []
    monkeypatch.setattr(chat_module, "CHAT_PROMPT", recorder)

    _single_turn(frozen)
    _multi_turn(frozen)

    assert len(_RecordingTemplate.calls) == 2, "each path renders one turn, so exactly two format calls"
    single_fields, multi_fields = _RecordingTemplate.calls
    assert set(single_fields) == set(multi_fields), (
        "the two paths supply different CHAT_PROMPT fields; a field added to the template "
        "would reach one path and raise KeyError on the other"
    )
    assert set(single_fields) == _template_fields(recorder), "every template field must be supplied"
    # The VALUE of comparison_instruction differs by design (the multi-entity
    # instruction); it is the field SET that must not drift between the paths.
    assert single_fields["comparison_instruction"] == ""
    assert "## Multi-entity comparison" in multi_fields["comparison_instruction"]


def test_chat_prompt_is_formatted_at_exactly_one_call_site():
    """The tail exists to leave one format site, so a new prompt field is a
    one-line change instead of one edit per retrieval path."""
    source = pathlib.Path(chat_module.__file__).read_text()
    assert source.count("CHAT_PROMPT.format(") == 1
    assert source.count("CHAT_USER_PROMPT.format(") == 1


# --- retrieval-side behaviour left per path ---------------------------------


def test_single_entity_gates_merged_list_then_slices_top_k(frozen):
    """Single-entity filters the MERGED list on the score gate and then slices
    ``[: k]``; with k=2 the faceted turn keeps only the two best of the three."""
    turn = _single_turn(frozen, dealtype="Seed")
    assert [s["id"] for s in turn.sources] == [1, 2]


def test_single_entity_drops_mid_score_article_when_unfaceted(frozen):
    """score 0.1 clears the faceted gate (0.0) but not the plain one (0.2)."""
    turn = _single_turn(frozen)
    assert [s["id"] for s in turn.sources] == [1, 3]


def test_multi_entity_gates_per_entity_without_a_top_k_slice(frozen):
    """Multi-entity filters PER ENTITY before dedupe and applies no ``[: k]``
    slice, so with k=2 the same three articles all survive. The two paths really
    do select differently; one shared gate would have changed this.
    """
    turn = _multi_turn(frozen, "comparison", dealtype="Seed")
    assert [s["id"] for s in turn.sources] == [1, 3, 2]


@pytest.mark.parametrize("path", ["single", "multi"])
def test_gate_to_empty_answers_without_calling_the_llm(frozen, path):
    """Both paths short-circuit on an empty gated set, with no note: the answer
    already explains the miss."""
    frozen.setattr(chat_module.config, "ASK_MIN_SCORE", 0.99)
    if path == "single":
        turn = _single_turn(frozen)
    else:
        turn = _multi_turn(frozen)
    assert turn.sources == []
    assert turn.answer == "No sufficiently relevant articles were found for this query."
    assert turn.note is None
    assert turn.needs_llm is False


def test_multi_entity_intersection_gated_to_empty_keeps_its_degradation_note(frozen):
    """The multi-entity early return is NOT the single-entity one: when an
    intersection has no article covering every entity it has already set the
    degradation note, and that note must survive the empty gated set. Sharing
    the two early returns would have hardcoded note=None and lost it."""
    frozen.setattr(chat_module.config, "ASK_MIN_SCORE", 0.99)
    turn = _multi_turn(frozen, "intersection")
    assert turn.sources == []
    assert turn.answer == "No sufficiently relevant articles were found for this query."
    assert turn.note == "No single article covers all of Alpha, Beta; showing related articles per entity."
    assert turn.needs_llm is False


def test_multi_entity_degraded_intersection_still_carries_its_note(frozen, monkeypatch):
    """When no article covers every entity the turn falls back to the union and
    says so. The single-entity path has no equivalent note, so the two early
    returns could not be shared as they stood."""
    pool = [
        [SourceArticle(id=10, title="Only Alpha", url="http://x/10", published_date="2024-01-01",
                       summary="Alpha only", body=BODIES[0], score=0.9)],
        [SourceArticle(id=11, title="Only Beta", url="http://x/11", published_date="2024-02-02",
                       summary="Beta only", body=BODIES[1], score=0.9)],
    ]
    calls = iter(pool)

    async def disjoint(retrieval_q, top_k, **kwargs):
        return (list(next(calls)), None, None, None)

    monkeypatch.setattr(main_module, "retrieve_with_auto_facet_fallback", disjoint)
    turn = _run(chat_module._prepare_multi_entity_turn(_multi("intersection"), QUESTION, list(HISTORY)))
    assert [s["id"] for s in turn.sources] == [10, 11]
    assert "No single article covers all of Alpha, Beta" in turn.note


# --- the shared body budget ------------------------------------------------


def test_body_char_limit_divides_the_total_across_the_sources(monkeypatch):
    """The budget both paths used to compute inline: as the source count grows
    each excerpt shrinks, so the total stays inside CHAT_TOTAL_BODY_CHARS."""
    monkeypatch.setattr(chat_module.config, "CHAT_BODY_CHAR_LIMIT", 500)
    monkeypatch.setattr(chat_module.config, "CHAT_TOTAL_BODY_CHARS", 900)
    assert chat_module._body_char_limit(1) == 500  # the per-article cap wins
    assert chat_module._body_char_limit(3) == 300  # 900 // 3
    assert chat_module._body_char_limit(0) == 500  # max(1, ...) keeps the division safe


@pytest.mark.parametrize("path", ["single", "multi"])
def test_both_paths_truncate_bodies_to_the_shared_budget(frozen, path):
    """A long article must come out cut to the same budget on both paths, with
    the cut marked by source_context's own truncation note."""
    frozen.setattr(chat_module.config, "CHAT_BODY_CHAR_LIMIT", 120)
    frozen.setattr(chat_module.config, "CHAT_TOTAL_BODY_CHARS", 100000)
    if path == "single":
        turn = _single_turn(frozen)
    else:
        turn = _multi_turn(frozen)
    note = main_module.BODY_TRUNCATION_NOTE
    assert note in turn.answer, "the excerpt must be cut, and the cut marked"
    assert BODIES[0] not in turn.answer, "the full body must not survive the budget"
    assert turn.answer.index(note) - turn.answer.index("Alpha body") == 120
