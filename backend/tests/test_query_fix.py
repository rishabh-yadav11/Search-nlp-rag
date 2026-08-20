"""QueryFixer tests: vocab loading (missing/corrupt artifacts), _build with
curated-entity counts, empty-vocab disable, _allowed_distance, the full fix()
pipeline (disabled/short/digit/known/no-suggestion/equal-input/low-count/
capitalization/fix-list branches), and init_fixer/fix_query no-ops.

symspellpy is faked in sys.modules so tests never build a real dictionary; the
fix() branches use a fake SymSpell-like object injected via __new__."""

import gzip
import json
import logging
import sys
import types

import pytest

import app.query_fix as query_fix_module
from app.query_fix import QueryFixer, _normalize_entity


def _write_vocab(tmp_path, rows, name="vocab.json.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(rows, f)
    return str(p)


def _fake_symspell(monkeypatch):
    calls = {"init": [], "create_dictionary_entry": []}

    class _FakeSymSpell:
        def __init__(self, max_dictionary_edit_distance, prefix_length):
            calls["init"].append((max_dictionary_edit_distance, prefix_length))

        def create_dictionary_entry(self, term, count):
            calls["create_dictionary_entry"].append((term, count))

    mod = types.ModuleType("symspellpy")
    mod.SymSpell = _FakeSymSpell
    monkeypatch.setitem(sys.modules, "symspellpy", mod)
    return calls


class _Sug:
    def __init__(self, term, distance, count):
        self.term = term
        self.distance = distance
        self.count = count


class _FakeSym:
    """SymSpell-like object returning per-word suggestion lists."""

    def __init__(self, suggestions=None):
        self.suggestions = suggestions or {}
        self.lookup_calls = []

    def lookup(self, word, verbosity=0, max_edit_distance=2):
        self.lookup_calls.append((word, verbosity, max_edit_distance))
        return self.suggestions.get(word, [])


def _fixer(sym=None, min_count=5, min_token_len=3):
    f = QueryFixer.__new__(QueryFixer)
    f._sym = sym
    f.min_count = min_count
    f.min_token_len = min_token_len
    f.max_edit = 2
    f._known = set()
    return f


# --- _normalize_entity ---


def test_normalize_entity_lowercases_strips_apostrophes_and_collapses_space():
    assert _normalize_entity("  ABC  Corp 'n' ") == "abc corp n"


# --- _load_vocab (lines 80-90) ---


def test_load_vocab_missing_path_returns_empty(tmp_path):
    assert QueryFixer._load_vocab(None) == {}
    assert QueryFixer._load_vocab(str(tmp_path / "nope.json.gz")) == {}


def test_load_vocab_corrupt_gzip_or_json_returns_empty(tmp_path, caplog):
    bad_gzip = tmp_path / "bad.json.gz"
    bad_gzip.write_bytes(b"not gzip data")
    bad_json = tmp_path / "bad2.json.gz"
    with gzip.open(bad_json, "wt", encoding="utf-8") as f:
        f.write("{not json")

    with caplog.at_level(logging.WARNING, logger="query_fix"):
        assert QueryFixer._load_vocab(str(bad_gzip)) == {}
        assert QueryFixer._load_vocab(str(bad_json)) == {}

    assert "query_fix: vocab load failed" in caplog.text


def test_load_vocab_filters_non_conforming_rows(tmp_path):
    vocab = _write_vocab(
        tmp_path,
        [["good", 5], ["", 5], [3, 5], ["bad-count", "x"], ["floaty", 7.5]],
    )
    assert QueryFixer._load_vocab(vocab) == {"good": 5, "floaty": 7}


# --- _build (lines 92-111) ---


def test_build_with_vocab_and_curated_entities(monkeypatch, tmp_path):
    calls = _fake_symspell(monkeypatch)
    vocab = _write_vocab(tmp_path, [["funding", 42], ["flipkart", 3], ["flip-kart", 7.5]])

    f = QueryFixer(vocab, entities=["Flipkart", "ABC Corp"])

    assert f._sym is not None
    assert f._known == {"funding", "flipkart", "flip-kart", "abc corp"}
    assert calls["init"] == [(2, 7)]
    # Curated entities get _CURATED_COUNT even when already in the vocab.
    assert calls["create_dictionary_entry"] == [
        ("funding", 42),
        ("flipkart", query_fix_module._CURATED_COUNT),
        ("flip-kart", 7),
        ("abc corp", query_fix_module._CURATED_COUNT),
    ]


def test_build_empty_vocab_disables_fixer(monkeypatch, tmp_path):
    calls = _fake_symspell(monkeypatch)
    vocab = _write_vocab(tmp_path, [])

    # __init__ always injects _NORMALIZED_ENTITIES when entities is empty, so
    # _build is called directly with an empty entity list to reach the
    # no-entries disabled branch.
    f = QueryFixer.__new__(QueryFixer)
    f.max_edit = 2
    f._sym = None
    f._known = set()
    f._build(vocab, entities=[])

    assert f._sym is None
    assert f._known == set()
    assert calls["init"] == []


def test_build_missing_vocab_without_entities_disables_fixer(monkeypatch, tmp_path):
    _fake_symspell(monkeypatch)
    f = QueryFixer.__new__(QueryFixer)
    f.max_edit = 2
    f._sym = None
    f._known = set()
    f._build(str(tmp_path / "absent.json.gz"), entities=[])
    assert f._sym is None


# --- _allowed_distance (line 113-115) ---


def test_allowed_distance_short_token_capped_at_one():
    f = _fixer(sym=None)
    assert f._allowed_distance("abc") == 1
    assert f._allowed_distance("abcd") == 1
    assert f._allowed_distance("abcde") == 2


# --- fix pipeline (lines 117-147) ---


def test_fix_disabled_or_empty_text_noop():
    f = _fixer(sym=None)
    assert f.fix("hello world") == ("hello world", [])

    enabled = _fixer(sym=_FakeSym())
    assert enabled.fix("") == ("", [])


def test_fix_skips_known_short_and_digit_tokens_without_lookup():
    sym = _FakeSym()
    f = _fixer(sym=sym)
    f._known = {"funding"}
    assert f.fix("funding ab 123") == ("funding ab 123", [])
    assert sym.lookup_calls == []


def test_fix_no_suggestion_passthrough():
    sym = _FakeSym()
    f = _fixer(sym=sym)
    assert f.fix("flipcart") == ("flipcart", [])
    assert sym.lookup_calls == [("flipcart", 1, 2)]


def test_fix_suggestion_equal_to_input_or_zero_distance_passthrough():
    same_term = _FakeSym({"flipcart": [_Sug("flipcart", 2, 100)]})
    assert _fixer(sym=same_term).fix("flipcart") == ("flipcart", [])

    zero_distance = _FakeSym({"flipcart": [_Sug("flipkart", 0, 100)]})
    assert _fixer(sym=zero_distance).fix("flipcart") == ("flipcart", [])


def test_fix_rejects_low_count_suggestion():
    sym = _FakeSym({"flipcart": [_Sug("flipkart", 2, 3)]})
    assert _fixer(sym=sym, min_count=5).fix("flipcart") == ("flipcart", [])


def test_fix_applies_correction_and_restores_capitalization():
    sym = _FakeSym({"flipcart": [_Sug("flipkart", 1, 500)]})
    text, fixes = _fixer(sym=sym).fix("Flipcart funding")

    assert text == "Flipkart funding"
    assert fixes == [{"token": "Flipcart", "correction": "Flipkart", "distance": 1, "count": 500}]


def test_fix_full_pipeline_multiple_fixes():
    sym = _FakeSym({
        "flipcart": [_Sug("flipkart", 1, 900)],
        "fundig": [_Sug("funding", 1, 800)],
    })
    text, fixes = _fixer(sym=sym).fix("flipcart fundig round")

    assert text == "flipkart funding round"
    assert fixes == [
        {"token": "flipcart", "correction": "flipkart", "distance": 1, "count": 900},
        {"token": "fundig", "correction": "funding", "distance": 1, "count": 800},
    ]


# --- init_fixer / fix_query (lines 150-173) ---


def test_init_fixer_disabled_sets_none():
    query_fix_module.init_fixer(False, None)
    assert query_fix_module.fixer is None


def test_init_fixer_enabled_builds_fixer(monkeypatch, tmp_path):
    _fake_symspell(monkeypatch)
    vocab = _write_vocab(tmp_path, [["funding", 42]])
    query_fix_module.init_fixer(True, vocab)
    try:
        assert query_fix_module.fixer is not None
        assert query_fix_module.fixer._sym is not None
    finally:
        query_fix_module.init_fixer(False, None)


def test_fix_query_noop_when_fixer_none():
    query_fix_module.init_fixer(False, None)
    assert query_fix_module.fix_query("flipcart") == ("flipcart", [])


def test_fix_query_delegates_to_fixer(monkeypatch, tmp_path):
    _fake_symspell(monkeypatch)
    vocab = _write_vocab(tmp_path, [["funding", 42]])
    query_fix_module.init_fixer(True, vocab)
    try:
        assert query_fix_module.fix_query("funding") == ("funding", [])
    finally:
        query_fix_module.init_fixer(False, None)