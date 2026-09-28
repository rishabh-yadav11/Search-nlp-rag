"""Unit tests for the internal (non-HTTP) pipeline functions in app.main:
embedding, intent rewrite, retrieval, body rescue/attach, and facet values."""

import asyncio
import logging
import math
from functools import partial

import pytest
from _common import make_point
from _support import OMIT, make_article
from _support import run_sync as _run
from qdrant_client.models import Fusion, FusionQuery, SparseVector

from app import main
from app.main import SourceArticle

# This file's copy left summary out of the constructor entirely; every other
# field was the shared default.
_article = partial(make_article, summary=OMIT)


class _Arr:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeDense:
    def __init__(self, vec):
        self.vec = vec
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return _Arr(self.vec)


class _SparseEmb:
    def __init__(self, indices, values):
        self.indices = _Arr(indices)
        self.values = _Arr(values)


class _FakeSparse:
    def __init__(self, indices, values):
        self.emb = _SparseEmb(indices, values)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return iter([self.emb])


class _Point:
    def __init__(self, id_, payload, score=1.0):
        self.id = id_
        self.payload = payload
        self.score = score


class _QueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    def __init__(self, points=None, scroll_pages=None):
        self.points = points or []
        self.retrieve_result = []
        self.query_points_calls = []
        self.retrieve_calls = []
        self.scroll_pages = scroll_pages or []
        self.scroll_calls = []

    async def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return _QueryResult(self.points)

    async def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return self.retrieve_result

    async def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        if self.scroll_pages:
            return self.scroll_pages.pop(0)
        return [], None


class _ProbeLock:
    def __init__(self):
        self.entries = 0
        self.exits = 0

    async def __aenter__(self):
        self.entries += 1
        return self

    async def __aexit__(self, *exc):
        self.exits += 1
        return False


class _FakeReranker:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self.logits



class _RecordingReranker:
    """Reranker fake that keeps the exact (query, document) pairs it was handed,
    so a test can assert both how many candidates entered the second pass and
    which ones they were -- not merely that predict() was called once."""

    def __init__(self, logits):
        self.logits = logits
        self.calls = 0
        self.pairs = []

    def predict(self, pairs):
        self.calls += 1
        self.pairs = list(pairs)
        return self.logits


class _FakeFacetClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _with_facet(qdrant: _FakeQdrant, pages):
    """Configure a fake Qdrant to return ``pages`` (list of (points, next_offset))
    from its public ``scroll`` method, used by _facet_values."""
    qdrant.scroll_pages = pages
    return qdrant


# --- _embed_sparse ---


def test_embed_sparse_returns_first_element():
    fake = _FakeSparse([1, 3], [0.9, 0.4])
    out = main._embed_sparse(fake, "query")
    assert fake.calls == 1
    assert out.indices.tolist() == [1, 3]
    assert out.values.tolist() == [0.9, 0.4]


# --- _effective_intent ---


def test_effective_intent_month_scoped_branch():
    rq, fd, td, dt, ind = main._effective_intent("top pharma deals of month january 2025", None, None)
    assert (rq, fd, td) == ("pharma deals", "2025-01-01", "2025-01-31")
    assert dt is None and ind is None


def test_effective_intent_user_dates_win():
    rq, fd, td, dt, ind = main._effective_intent("deals in 2025", "2024-01-01", "2024-12-31")
    assert (rq, fd, td) == ("deals in 2025", "2024-01-01", "2024-12-31")
    assert dt is None and ind is None


def test_effective_intent_no_intent_passthrough():
    rq, fd, td, dt, ind = main._effective_intent("latest deals", None, None)
    # No category facet; the recency term 'latest' is stripped for retrieval
    # (the recency intent still drives ranking weight separately).
    assert (rq, fd, td) == ("deals", None, None)
    assert dt is None and ind is None


def test_effective_intent_normalizes_word_numbers():
    """'top ten ipo' must retrieve exactly like 'top 10 ipo': the literal word
    'ten' would otherwise dilute the embedding/rerank match against titles like
    'Ten Sports'."""
    assert main._effective_intent("top ten ipo", None, None) == ("top 10 ipo", None, None, None, None)
    assert main._effective_intent("top ten deals", None, None) == ("top 10 deals", None, None, None, None)
    assert main._effective_intent("top ten ipo", "2024-01-01", "2024-12-31") == (
        "top 10 ipo", "2024-01-01", "2024-12-31", None, None,
    )


# --- _retrieval_queries ---


def test_retrieval_queries_year_in_review_two_legs():
    assert main._retrieval_queries("top 3 unicorns created in 2025") == [
        "Flashback 2025 unicorns created",
        "unicorns created",
    ]


def test_retrieval_queries_month_scoped_single_leg():
    assert main._retrieval_queries("top pharma deals of month january 2025") == ["pharma deals"]


def test_retrieval_queries_plain_single():
    assert main._retrieval_queries("latest deals") == ["latest deals"]


def test_retrieval_queries_dedup_when_flashback_equals_topic():
    assert main._retrieval_queries("top deals") == ["top deals"]


# --- hybrid_search ---


def test_hybrid_search_cache_miss(monkeypatch, fake_cache):
    cache = fake_cache()
    monkeypatch.setattr(main, "cache", cache)
    dense = _FakeDense([0.1, 0.2, 0.3])
    sparse = _FakeSparse([1, 3], [0.9, 0.4])
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    qdrant = _FakeQdrant(
        points=[
            _Point(
                1,
                {
                    "title": "T1",
                    "url": "u1",
                    "summary": "s1",
                    "body": "b1",
                    "author_names": ["A"],
                    "industry_names": ["Fin"],
                    "dealtype_names": ["M&A"],
                },
                score=0.8,
            )
        ]
    )
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "EMBED_MODEL", "dense-model")
    monkeypatch.setattr(main.config, "SPARSE_MODEL", "sparse-model")
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    monkeypatch.setattr(main.config, "VECTOR_CACHE_TTL_SECONDS", 123)

    articles = _run(main.hybrid_search("query", 8))

    assert dense.calls == 1
    assert sparse.calls == 1
    assert len(cache.sets) == 1
    key, value, ttl = cache.sets[0]
    assert key == "vec:dense-model|sparse-model:query"
    assert ttl == 123
    assert value["dense"] == [0.1, 0.2, 0.3]
    assert value["si"] == [1, 3]
    assert value["sv"] == [0.9, 0.4]
    assert [a.id for a in articles] == [1]
    assert articles[0].title == "T1"
    assert articles[0].body == "b1"
    assert articles[0].author_names == ["A"]

    kwargs = qdrant.query_points_calls[0]
    assert kwargs["collection_name"] == "col"
    assert kwargs["limit"] == 8
    assert kwargs["with_payload"] == main._PAYLOAD_FIELDS
    assert kwargs["query_filter"] is None
    assert isinstance(kwargs["query"], FusionQuery)
    assert kwargs["query"].fusion == Fusion.RRF
    assert len(kwargs["prefetch"]) == 2
    assert kwargs["prefetch"][0].using == "dense"
    assert kwargs["prefetch"][1].using == "sparse"
    assert kwargs["prefetch"][0].limit == 32


def test_hybrid_search_cache_hit_skips_encoding(monkeypatch, fake_cache):
    vec = {"dense": [0.1, 0.2], "si": [1], "sv": [0.7]}
    monkeypatch.setattr(main, "cache", fake_cache(get_result=vec))
    dense = _FakeDense([9.9, 9.9])
    sparse = _FakeSparse([9], [9.9])
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    qdrant = _FakeQdrant(points=[_Point(2, {"title": "T2", "url": "u2"}, score=0.5)])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    articles = _run(main.hybrid_search("query", 4, with_body=True))

    assert dense.calls == 0
    assert sparse.calls == 0
    assert [a.id for a in articles] == [2]
    kwargs = qdrant.query_points_calls[0]
    assert kwargs["with_payload"] is True
    assert kwargs["query"].fusion == Fusion.RRF
    assert isinstance(kwargs["prefetch"][1].query, SparseVector)


def test_hybrid_search_acquires_inference_lock_once_on_miss(monkeypatch, fake_cache):
    monkeypatch.setattr(main, "cache", fake_cache())
    monkeypatch.setitem(main.state, "model", _FakeDense([0.1]))
    monkeypatch.setitem(main.state, "sparse_model", _FakeSparse([0], [0.5]))
    monkeypatch.setitem(main.state, "qdrant", _FakeQdrant(points=[_Point(1, {"title": "T", "url": "u"})]))
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    lock = _ProbeLock()
    monkeypatch.setattr(main, "inference_lock", lock)

    _run(main.hybrid_search("query", 4))

    assert lock.entries == 1
    assert lock.exits == 1


def test_hybrid_search_skips_null_and_empty_payload_points(monkeypatch, fake_cache):
    monkeypatch.setattr(main, "cache", fake_cache())
    monkeypatch.setitem(main.state, "model", _FakeDense([0.1]))
    monkeypatch.setitem(main.state, "sparse_model", _FakeSparse([0], [0.5]))
    monkeypatch.setitem(
        main.state,
        "qdrant",
        _FakeQdrant(
            points=[
                _Point(1, {"title": "T1", "url": "u1", "summary": "s1"}),
                _Point(2, None),
                _Point(3, {}),
            ]
        ),
    )
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    articles = _run(main.hybrid_search("query", 8))

    assert [a.id for a in articles] == [1]


def test_hybrid_search_surfaces_the_stored_content_type(monkeypatch, fake_cache):
    """A payload written by the real writer reaches SourceArticle.content_type.

    This is the read half of the feature that was dead end to end: the write
    side (make_point) and the read side (hybrid_search -> SourceArticle) are
    joined here through a real payload, so a break on either side is caught by
    one test. An empty stored value must read back as None, matching the
    `or None` mapping, so articles with no content type stay indistinguishable
    from articles that were never backfilled.
    """
    monkeypatch.setattr(main, "cache", fake_cache())
    monkeypatch.setitem(main.state, "model", _FakeDense([0.1, 0.2]))
    monkeypatch.setitem(main.state, "sparse_model", _FakeSparse([1], [0.5]))
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    def stored_payload(content_type):
        return make_point(
            {
                "id": 7,
                "title": "T7",
                "url": "u7",
                "summary": "s7",
                "body": "b7",
                "published_date": "2025-06-01T00:00:00",
                "category": "Series A",
                "content_type": content_type,
                "author_names": ["A"],
                "industry_names": ["Fin"],
                "dealtype_names": ["M&A"],
            },
            _Arr([0.1, 0.2]),
            _SparseEmb([1], [0.5]),
        ).payload

    qdrant = _FakeQdrant(
        points=[
            _Point(7, stored_payload("Interview"), score=0.9),
            _Point(8, stored_payload(""), score=0.8),
        ],
    )
    monkeypatch.setitem(main.state, "qdrant", qdrant)

    articles = _run(main.hybrid_search("query", 8))

    assert [(a.id, a.content_type) for a in articles] == [(7, "Interview"), (8, None)]
    # The field must actually be requested from Qdrant, not merely mapped.
    assert "content_type" in qdrant.query_points_calls[0]["with_payload"]


# --- retrieve_by_date_window (date-only fallback fillers) ---


def _date_window_qdrant(monkeypatch):
    """A scroll page of two window articles, newest first, as Qdrant returns
    them under the published_date order_by."""
    page = (
        [
            _Point(11, {"title": "Newer", "published_date": "2025-06-02T00:00:00"}),
            _Point(12, {"title": "Older", "published_date": "2025-05-02T00:00:00"}),
        ],
        None,
    )
    # Two copies: the knob tests call the real retrieval twice, and the fake
    # hands out one page per call.
    qdrant = _FakeQdrant(scroll_pages=[page, page])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    return qdrant


def _date_window_articles():
    return _run(main.retrieve_by_date_window(top_k=5, from_date="2025-05-01", to_date="2025-06-30"))


def test_date_fillers_take_their_own_knob(monkeypatch):
    """The relevance floor handed to date-only fillers is its own knob, read
    where the articles are built: retuning it must move the fillers, and
    nothing else. Pinned to the shipped 0.2 first so a developer .env cannot
    decide the starting expectation (the default itself is asserted against a
    clean parse below)."""
    _date_window_qdrant(monkeypatch)
    monkeypatch.setattr(main.config, "DATE_FILLER_SCORE", 0.2)
    assert [a.score for a in _date_window_articles()] == [0.2, 0.2]

    monkeypatch.setattr(main.config, "DATE_FILLER_SCORE", 0.05)
    assert [a.score for a in _date_window_articles()] == [0.05, 0.05]


def test_date_fillers_do_not_follow_the_inclusion_gate_at_the_call_site(monkeypatch):
    """Regression (issue #300), call-site half: the floor must not be
    re-derived from config.ASK_MIN_SCORE where the articles are built, which
    would drag every date-only filler up the moment an operator retuned the
    gate. The import-time capture the issue found is what
    test_date_fillers_take_their_own_knob pins (a module-level constant bound
    before the test runs cannot follow a monkeypatched knob at all)."""
    _date_window_qdrant(monkeypatch)
    monkeypatch.setattr(main.config, "DATE_FILLER_SCORE", 0.2)
    monkeypatch.setattr(main.config, "ASK_MIN_SCORE", 0.9)
    assert [a.score for a in _date_window_articles()] == [0.2, 0.2]


def test_date_filler_knob_ships_the_value_the_alias_resolved_to(parse_config):
    """Default configuration must be byte-identical to the pre-#300 behaviour:
    _DATE_FILLER_SCORE = config.ASK_MIN_SCORE resolved to ASK_MIN_SCORE's own
    0.2 default out of the box, and the two shipped defaults stay equal.

    The ordering assertion is about the shipped PAIR, not about a coupling
    between the knobs: chat filters sources with `score >= ASK_MIN_SCORE`, so a
    filler floor below the gate would be dropped before the model ever saw it
    and the temporal fallback would silently stop working. The two are
    independent now, which means raising one in a deployment's .env without
    raising the other is a real (documented) way to break that -- but the
    defaults this branch ships must not start out broken.
    """
    shipped = parse_config()
    assert shipped.ASK_MIN_SCORE == 0.2
    assert shipped.DATE_FILLER_SCORE == 0.2
    assert shipped.DATE_FILLER_SCORE >= shipped.ASK_MIN_SCORE


def test_retuning_the_inclusion_gate_does_not_move_the_date_filler_floor(parse_config):
    """The deployment-level half of the #300 regression. A shipped .env that
    raises ASK_MIN_SCORE to 0.9 must not drag the date-only fillers up with
    it: with the old import-time alias the filler floor followed the gate into
    every window query. Parsed from `ASK_MIN_SCORE=0.9` and nothing else, so
    the result cannot be an echo of this machine's own .env."""
    parsed = parse_config(ASK_MIN_SCORE="0.9")
    assert parsed.ASK_MIN_SCORE == 0.9
    assert parsed.DATE_FILLER_SCORE == 0.2


def test_date_filler_floor_is_tunable_from_the_environment(parse_config):
    """...and it is a real knob, not a constant with a new name: the shipped
    template's value is what the app parses when an operator sets it."""
    assert parse_config(DATE_FILLER_SCORE="0.45").DATE_FILLER_SCORE == 0.45


def _filler_gate_warnings(caplog):
    return [r for r in caplog.records if "DATE_FILLER_SCORE" in r.getMessage()]


def test_a_filler_floor_below_the_chat_gate_is_logged_at_startup(parse_config, caplog):
    """The two knobs are independent, which makes the mis-ordered pairing
    reachable from a deployment's .env — and it fails silently: every date-only
    filler is dropped by the chat gate, the temporal fallback contributes
    nothing, and the turn answers "no relevant articles" with no other trace.
    config.py warns about it, the way it warns about clamped knobs, instead of
    raising: the service still serves, but the misconfiguration is named."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config(ASK_MIN_SCORE="0.6")
    warnings = _filler_gate_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "0.2" in message and "0.6" in message  # both knob values, as configured
    assert "temporal fallback" in message  # and the consequence


def test_the_filler_floor_warning_stays_quiet_for_sound_configurations(parse_config, caplog):
    """A warning that fires for the shipped defaults is noise nobody reads, and
    one that fires for a correctly raised pair trains the reader to ignore it.
    Only the mis-ordered combination may speak."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config()
    assert _filler_gate_warnings(caplog) == []

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # Gate raised and the floor raised with it: a valid deployment.
        parse_config(ASK_MIN_SCORE="0.6", DATE_FILLER_SCORE="0.7")
    assert _filler_gate_warnings(caplog) == []

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config(ASK_MIN_SCORE="0.6", DATE_FILLER_SCORE="0.6")
    assert _filler_gate_warnings(caplog) == []  # equal is fine: the gate is `>=`


# --- body_rescue ---


def test_body_rescue_empty_articles(monkeypatch):
    fake = _FakeReranker([1.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    assert _run(main.body_rescue("query", [])) == []
    assert fake.calls == 0


def test_body_rescue_skips_when_top_score_strong(monkeypatch):
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a = _article(1, 0.5, body="has body")
    out = _run(main.body_rescue("some query", [a]))
    assert fake.calls == 0
    assert out[0].score == 0.5


def test_body_rescue_skips_stopword_only_query(monkeypatch):
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a = _article(1, 0.1, body="has body")
    out = _run(main.body_rescue("a the of and", [a]))
    assert fake.calls == 0
    assert out[0].score == 0.1


def test_body_rescue_skips_when_bodies_empty(monkeypatch):
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a1 = _article(1, 0.1)
    a2 = _article(2, 0.1)
    out = _run(main.body_rescue("funding deals", [a1, a2]))
    assert fake.calls == 0
    assert [x.id for x in out] == [1, 2]


def test_body_rescue_lifts_deep_body_match_and_reorders(monkeypatch):
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)
    fake = _FakeReranker([5.0, -2.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    matching = _article(1, 0.1, body=("filler " * 100) + ("2008 crisis central banks lessons " * 40))
    unrelated = _article(2, 0.1, body=("weather and markets " * 200))
    out = _run(main.body_rescue("lessons 2008 crisis central banks", [matching, unrelated]))
    assert fake.calls == 1
    assert [a.id for a in out] == [1, 2]
    assert abs(out[0].score - 1.0 / (1.0 + math.exp(-5.0))) < 1e-9
    assert abs(out[1].score - max(0.1, 1.0 / (1.0 + math.exp(2.0)))) < 1e-9


def test_body_rescue_reranker_error_propagates(monkeypatch):
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)

    class _Boom:
        def predict(self, pairs):
            raise RuntimeError("model load failed")

    monkeypatch.setitem(main.state, "reranker", _Boom())
    a = _article(1, 0.1, body="some body with funding deals")
    with pytest.raises(RuntimeError):
        _run(main.body_rescue("funding deals", [a]))


def test_body_rescue_makes_no_second_pass_when_gate_is_off(monkeypatch):
    """The gate has to live in body_rescue itself, because body_rescue is what
    pays for the second cross-encoder pass. With ENABLE_BODY_RESCUE off, weak
    results that do have bodies must never reach the reranker -- otherwise a
    caller that forgets the check silently pays the cost again."""
    monkeypatch.setattr(main.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)
    fake = _RecordingReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    arts = [
        _article(1, 0.1, body="funding deals round " * 20),
        _article(2, 0.05, body="funding deals round " * 20),
    ]
    out = _run(main.body_rescue("funding deals", arts))
    assert fake.calls == 0
    assert fake.pairs == []
    assert [a.score for a in out] == [0.1, 0.05]


def test_body_rescue_caps_candidates_entering_the_second_pass(monkeypatch):
    """The rescue costs one cross-encoder prediction per candidate, so chat
    handing it CHAT_MAX_SOURCES articles must not mean 20 predictions under
    the global inference lock."""
    monkeypatch.setattr(main.config, "ENABLE_BODY_RESCUE", True)
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)
    monkeypatch.setattr(main.config, "BODY_RESCUE_MAX_CANDIDATES", 3)
    fake = _RecordingReranker([5.0] * 9)
    monkeypatch.setitem(main.state, "reranker", fake)
    arts = [_article(i, 0.1, body="funding deals round " * 20) for i in range(1, 10)]
    _run(main.body_rescue("funding deals", arts))
    assert fake.calls == 1
    assert len(fake.pairs) == 3


def test_body_rescue_shortlists_by_body_overlap_not_by_score(monkeypatch):
    """A weak title+summary score is precisely the reason the rescue exists:
    the article worth rescoring is the one whose match lives in the body. So
    the limited budget goes to the highest body-window overlap -- not to the
    top scorer, and not to the first articles in list order (the buried match
    here is second)."""
    monkeypatch.setattr(main.config, "ENABLE_BODY_RESCUE", True)
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.95)
    monkeypatch.setattr(main.config, "BODY_RESCUE_MAX_CANDIDATES", 1)
    fake = _RecordingReranker([5.0, 5.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    top_scored = _article(1, 0.9, title="Strong on title", body="weather and markets " * 100)
    buried = _article(2, 0.05, title="Buried match", body="lessons 2008 crisis central banks " * 40)
    _run(main.body_rescue("lessons 2008 crisis central banks", [top_scored, buried]))
    assert len(fake.pairs) == 1
    assert "Buried match" in fake.pairs[0][1]
    assert "Strong on title" not in fake.pairs[0][1]


# --- _attach_bodies ---


def test_attach_bodies_empty_skips_retrieve(monkeypatch):
    qdrant = _FakeQdrant()
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    _run(main._attach_bodies([]))
    assert qdrant.retrieve_calls == []


def test_attach_bodies_sets_payloads_for_found_ids(monkeypatch):
    qdrant = _FakeQdrant()
    qdrant.retrieve_result = [_Point(1, {"body": "body1"}), _Point(3, {"body": "body3"})]
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    arts = [_article(1, 0.5), _article(2, 0.5), _article(3, 0.5)]
    _run(main._attach_bodies(arts))
    assert qdrant.retrieve_calls[0]["collection_name"] == "col"
    assert qdrant.retrieve_calls[0]["ids"] == [1, 2, 3]
    assert qdrant.retrieve_calls[0]["with_payload"] == ["body"]
    assert arts[0].body == "body1"
    assert arts[1].body == ""
    assert arts[2].body == "body3"


def test_attach_bodies_none_payload_gets_empty_body(monkeypatch):
    qdrant = _FakeQdrant()
    qdrant.retrieve_result = [_Point(1, None)]
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    arts = [_article(1, 0.5)]
    _run(main._attach_bodies(arts))
    assert arts[0].body == ""


def test_attach_bodies_matches_string_point_id_to_int_article_id(monkeypatch):
    # Qdrant may return a string point id; it must still attach the body to the
    # int-id article rather than silently dropping body context.
    qdrant = _FakeQdrant()
    qdrant.retrieve_result = [_Point("1", {"body": "body1"}), _Point("3", {"body": "body3"})]
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    arts = [_article(1, 0.5), _article(2, 0.5), _article(3, 0.5)]
    _run(main._attach_bodies(arts))
    assert arts[0].body == "body1"
    assert arts[1].body == ""
    assert arts[2].body == "body3"


def test_attach_bodies_retrieve_error_propagates(monkeypatch):
    class _Boom:
        async def retrieve(self, **kwargs):
            raise RuntimeError("qdrant down")

    monkeypatch.setitem(main.state, "qdrant", _Boom())
    with pytest.raises(RuntimeError):
        _run(main._attach_bodies([_article(1, 0.5)]))


# --- _retrieval_leg ---


def test_retrieval_leg_expands_query_and_uses_rerank_candidates(monkeypatch):
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["qfilter"] = qfilter
        return []

    monkeypatch.setattr(main, "expand_query", lambda q: "EXPANDED " + q)
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main.config, "ENABLE_QUERY_EXPANSION", True)
    monkeypatch.setattr(main.config, "RERANK_CANDIDATES", 12)

    _run(main._retrieval_leg("funding deals", 5, None))

    assert captured["query"] == "EXPANDED funding deals"
    assert captured["top_k"] == 12
    assert captured["qfilter"] is None


def test_retrieval_leg_skips_expansion_for_flashback(monkeypatch):
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["query"] = query
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(main, "expand_query", lambda q: "EXPANDED " + q)
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main.config, "ENABLE_QUERY_EXPANSION", True)
    monkeypatch.setattr(main.config, "RERANK_CANDIDATES", 12)

    _run(main._retrieval_leg("Flashback 2025 deals", 5, None))

    assert captured["query"] == "Flashback 2025 deals"


def test_retrieval_leg_no_expansion_when_disabled(monkeypatch):
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["query"] = query
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(main, "expand_query", lambda q: "EXPANDED " + q)
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main.config, "ENABLE_QUERY_EXPANSION", False)
    monkeypatch.setattr(main.config, "RERANK_CANDIDATES", 12)

    _run(main._retrieval_leg("funding deals", 20, None))

    assert captured["query"] == "funding deals"
    assert captured["top_k"] == 20


# --- source_context ---


def test_source_context_all_facets_and_body():
    a = SourceArticle(
        id=1,
        title="T",
        url="u",
        published_date="2024-01-01",
        summary="sum",
        body="body text",
        author_names=["Alice", "Bob"],
        industry_names=["Fintech"],
        dealtype_names=["Funding"],
        score=0.5,
    )
    out = main.source_context(a, 1)
    assert "[1] T (2024-01-01 | Authors: Alice, Bob | Industry: Fintech | Dealtype: Funding)" in out
    assert "sum" in out
    assert "body text" in out


def test_source_context_no_facets_no_summary():
    a = SourceArticle(id=2, title="T", url="u", published_date=None, score=0.5)
    out = main.source_context(a, 3)
    assert out == "[3] T (n/a)"


def test_source_context_truncates_body_to_body_limit():
    a = SourceArticle(id=1, title="T", url="u", summary="s", body="x" * 100, score=0.5)
    out = main.source_context(a, 1, body_limit=20)
    # Everything after the "[1] T (n/a)" meta line and the "s" summary.
    excerpt = out.split("\n", 2)[2]
    # The cap bounds the body characters; the note marks the cut so the model
    # can tell a capped excerpt from an article that simply ended.
    assert excerpt == "x" * 20 + main.BODY_TRUNCATION_NOTE


# --- _facet_values ---


def test_facet_values_sorted_and_scroll_kwargs(monkeypatch):
    points = [
        _Point(1, {"industry_names": "Finance"}),
        _Point(2, {"industry_names": "TMT"}),
        _Point(3, {"industry_names": "General"}),
    ]
    qdrant = _with_facet(_FakeQdrant(), [(points, None)])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    out = _run(main._facet_values("industry_names"))

    assert out == ["Finance", "General", "TMT"]
    assert qdrant.scroll_calls[0]["collection_name"] == "col"
    assert qdrant.scroll_calls[0]["with_payload"] == ["industry_names"]


def test_facet_values_filters_non_string_hits(monkeypatch):
    points = [
        _Point(1, {"industry_names": "Finance"}),
        _Point(2, {"industry_names": 42}),
        _Point(3, {"industry_names": None}),
    ]
    qdrant = _with_facet(_FakeQdrant(), [(points, None)])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    assert _run(main._facet_values("industry_names")) == ["Finance"]


def test_facet_values_empty_result(monkeypatch):
    qdrant = _with_facet(_FakeQdrant(), [([], None)])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    assert _run(main._facet_values("industry_names")) == []


def test_facet_values_error_propagates(monkeypatch):
    qdrant = _FakeQdrant()

    async def _boom(**kwargs):
        raise RuntimeError("qdrant down")

    qdrant.scroll = _boom
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    with pytest.raises(RuntimeError):
        _run(main._facet_values("industry_names"))


# --- _best_body_window tail branch ---


def test_best_body_window_tail_wins_on_tie():
    body = ("filler " * 60) + "alpha beta gamma"
    tokens = {"alpha", "beta", "gamma"}
    out = main._best_body_window(body, tokens, 50, 50)
    assert out == body[-50:]


# --- _best_body_window window budget ---


def test_effective_step_leaves_the_default_scan_alone():
    """The default scan must be bit-for-bit unchanged by the budget.

    A full 50,000-char body at win=1500/step=500 is 98 window starts,
    comfortably inside the default budget of 200, so no operator upgrading
    this branch sees a different body scan.
    """
    positions = 50_000 - 1500 + 1
    assert main._effective_step(positions, 500, 200) == 500
    assert -(-positions // main._effective_step(positions, 500, 200)) == 98


def test_effective_step_widens_a_legal_but_expensive_stride():
    """Clamping BODY_RESCUE_STEP alone does not bound the work: step=1 is
    inside the clamp and still scores 48,501 windows of a 50K body (~117ms,
    and body_rescue scans every body-bearing article before the candidate cap
    applies, so 20 articles cost ~2.3s for one chat turn).
    """
    positions = 49_700 - 1500 + 1
    widened = main._effective_step(positions, 1, 200)
    assert widened > 1
    assert -(-positions // widened) <= 200


def test_effective_step_never_returns_a_zero_stride():
    """`range(0, n, 0)` raises ValueError. The config clamp already rejects 0,
    but `_best_body_window` is module-level and directly callable, so it must
    not depend on every caller having gone through config."""
    # A zero step is floored to 1 and then still widened to fit the budget;
    # what matters is that the returned stride is always usable by range().
    widened = main._effective_step(1000, 0, 200)
    assert widened >= 1
    assert -(-1000 // widened) <= 200
    # A non-positive budget disables widening rather than producing stride 0.
    assert main._effective_step(1000, 0, 0) == 1
    out = main._best_body_window("filler " * 3000, {"alpha"}, 1500, 0)
    assert isinstance(out, str) and out


def test_best_body_window_finds_dense_region_at_the_default_budget():
    """The budget must be free at any value that does not force a widening.

    This is the case operators actually run: win=1500/step=500 over this body
    is 98 windows against a budget of 200, so the stride never moves and the
    dense region is still found.
    """
    body = ("filler " * 4000) + ("alpha beta gamma " * 40) + ("filler " * 2000)
    out = main._best_body_window(body, {"alpha", "beta", "gamma"}, 1500, 500, max_windows=200)
    low = out.lower()
    assert "alpha" in low and "gamma" in low


def test_best_body_window_a_tight_budget_can_straddle_the_dense_region():
    """The trade-off, stated rather than papered over: a stride coarse enough
    to widen can skip a token-dense region. That is the cost of capping the
    work, and it is only reachable when a budget tighter than the chosen
    stride needs is configured -- the default budget never widens.

    A budget of 1 forces the single window at start=0, so the result is
    deterministically `body[:1500]` and cannot depend on window arithmetic.
    """
    body = ("filler " * 4000) + ("alpha beta gamma " * 40) + ("filler " * 2000)
    out = main._best_body_window(body, {"alpha", "beta", "gamma"}, 1500, 1, max_windows=1)
    assert out == body[:1500]
    # The same call with no budget scans everything and finds the region,
    # which is exactly what the budget gives up.
    unbudgeted = main._best_body_window(body, {"alpha", "beta", "gamma"}, 1500, 1)
    assert "alpha" in unbudgeted.lower()


class _IterationCountingTokens(set):
    """A token set that records how many times the scan loop iterated it.

    `_best_body_window` scores each window with `sum(1 for t in tokens ...)`,
    so the number of times the set is iterated is the number of windows scored
    (plus the single tail comparison the helper always makes). This observes
    the real loop through `_best_body_window` itself, which is the only place
    the budget is actually applied.
    """

    def __init__(self, items):
        super().__init__(items)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()

    def windows_scored(self):
        return self.iterations - 1


def test_best_body_window_applies_the_window_budget():
    """The budget must be enforced by `_best_body_window`, not merely be
    available on `_effective_step`.

    Asserting only the helper leaves the single line that applies it
    (`step = _effective_step(...)`) uncovered: deleting that line leaves the
    whole suite green while the DoS bound silently disappears.
    """
    body = "filler " * 7100  # 49,700 chars
    tokens = _IterationCountingTokens({"alpha", "beta", "gamma"})
    main._best_body_window(body, tokens, 1500, 1, max_windows=200)
    # Unbudgeted, step=1 would score 48,201 windows.
    assert tokens.windows_scored() <= 200, tokens.windows_scored()


def test_best_body_window_does_not_widen_a_default_scan():
    """The default scan already fits the budget, so the stride must not move
    and the default rescue must score exactly the same windows as before."""
    body = "filler " * 7100
    tokens = _IterationCountingTokens({"alpha", "beta", "gamma"})
    main._best_body_window(body, tokens, 1500, 500, max_windows=200)
    assert tokens.windows_scored() == 97  # ceil(48201 / 500)




# --- lifespan ---


def _stub_lifespan_deps(monkeypatch, chat_connect_error=None, auth_connect_error=None):
    """Fake every startup dependency of app.main.lifespan. Returns the fakes so
    tests can assert on their lifecycle state."""

    class FakeChatStore:
        def __init__(self):
            self.closed = False

        async def connect(self):
            if chat_connect_error:
                raise chat_connect_error
            self.connected = True

        async def close(self):
            self.closed = True

    class FakeAuthStore:
        def __init__(self):
            self.closed = False

        async def connect(self):
            if auth_connect_error:
                raise auth_connect_error
            self.connected = True

        async def close(self):
            self.closed = True

    class FakeQdrant:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    class FakeCache:
        def __init__(self):
            self.closed = False

        async def get(self, key):
            return None

        async def set(self, key, value, ttl=None):
            pass

        async def close(self):
            self.closed = True

    chat_store = FakeChatStore()
    auth_store = FakeAuthStore()
    qdrant = FakeQdrant()
    cache = FakeCache()

    monkeypatch.setattr(main, "DenseEncoder", lambda *a, **k: object())
    monkeypatch.setattr(main, "SparseTextEmbedding", lambda *a, **k: object())
    monkeypatch.setattr(main, "Reranker", lambda *a, **k: object())
    monkeypatch.setattr(main, "AsyncQdrantClient", lambda *a, **k: qdrant)
    monkeypatch.setattr(main, "AsyncOpenAI", lambda *a, **k: object())
    monkeypatch.setattr(main, "cache", cache)

    monkeypatch.setattr(main.chat_module, "ChatStore", lambda *a, **k: chat_store)
    monkeypatch.setattr(main.chat_module, "store", None)

    async def _retention_loop():
        await asyncio.sleep(3600)

    monkeypatch.setattr(main.chat_module, "retention_loop", _retention_loop)

    monkeypatch.setattr(main.auth_module, "AuthStore", lambda *a, **k: auth_store)
    monkeypatch.setattr(main.auth_module, "store", None)

    async def _bootstrap_admin():
        return None

    monkeypatch.setattr(main.auth_module, "bootstrap_admin", _bootstrap_admin)

    fixer_calls = []
    monkeypatch.setattr(main, "init_fixer", lambda *a, **k: fixer_calls.append((a, k)))

    closed = []

    async def _close_analytics():
        closed.append("analytics")

    async def _close_cost():
        closed.append("cost")

    monkeypatch.setattr(main, "close_analytics", _close_analytics)
    monkeypatch.setattr(main, "close_cost_budget", _close_cost)

    return {
        "chat_store": chat_store,
        "auth_store": auth_store,
        "qdrant": qdrant,
        "cache": cache,
        "fixer_calls": fixer_calls,
        "closed": closed,
    }


def _restore_state(orig):
    main.state.clear()
    main.state.update(orig)


def test_lifespan_startup_and_teardown(monkeypatch):
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch)

    async def scenario():
        async with main.lifespan(None):
            assert main.state["model"] is not None
            assert main.state["sparse_model"] is not None
            assert main.state["reranker"] is not None
            assert main.state["qdrant"] is deps["qdrant"]
            assert main.state["llm"] is not None
            assert main.chat_module.store is deps["chat_store"]
            assert main.auth_module.store is deps["auth_store"]
            assert deps["fixer_calls"][0][1]["max_edit"] == main.config.QUERY_FIX_MAX_EDIT
            assert "chat_retention" in main.state

    try:
        _run(scenario())
    finally:
        _restore_state(orig)

    assert deps["chat_store"].closed is True
    assert deps["auth_store"].closed is True
    assert deps["qdrant"].closed is True
    assert deps["cache"].closed is True
    assert deps["closed"] == ["analytics", "cost"]


def test_lifespan_llm_none_without_api_key(monkeypatch):
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "")
    deps = _stub_lifespan_deps(monkeypatch)

    async def scenario():
        async with main.lifespan(None):
            assert main.state["llm"] is None

    try:
        _run(scenario())
    finally:
        _restore_state(orig)

    assert deps["chat_store"].closed is True


def test_lifespan_logs_a_placeholder_gemini_key_by_name(monkeypatch, caplog):
    """The startup log is the whole point of not crashing on a bad key: an
    operator reading a chat-broken deploy finds the cause in one line instead
    of a per-turn 401, and the process stays up to serve /health and /ready.
    So the lifespan must actually emit it -- which is what this test holds."""
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "your_key_here")
    _stub_lifespan_deps(monkeypatch)

    async def scenario():
        async with main.lifespan(None):
            pass

    try:
        with caplog.at_level(logging.ERROR, logger="health"):
            _run(scenario())
    finally:
        _restore_state(orig)

    assert "GEMINI_API_KEY" in caplog.text
    assert "placeholder" in caplog.text
    assert "your_key_here" not in caplog.text, "the log must name the fault, never print the key"


def test_lifespan_logs_nothing_for_a_usable_gemini_key(monkeypatch, caplog):
    """The opposite guard: an ERROR line on every healthy boot trains operators
    to ignore the one that matters."""
    orig = dict(main.state)
    # A real key's shape, assembled rather than written out: a credential-shaped
    # literal on a line naming GEMINI_API_KEY is what a secrets scanner flags
    # (same reason, and the same fixture, as REAL_GEMINI_KEY in test_health.py).
    real_key = "AI" + "za" + "SyD-Example_Key" + "0123456789" + "abcdefghij"
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", real_key)
    _stub_lifespan_deps(monkeypatch)

    async def scenario():
        async with main.lifespan(None):
            pass

    try:
        with caplog.at_level(logging.ERROR, logger="health"):
            _run(scenario())
    finally:
        _restore_state(orig)

    assert "GEMINI_API_KEY" not in caplog.text


def test_lifespan_startup_failure_propagates(monkeypatch):
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch, chat_connect_error=RuntimeError("sqlite locked"))

    async def scenario():
        async with main.lifespan(None):
            pass

    try:
        with pytest.raises(RuntimeError):
            _run(scenario())
    finally:
        _restore_state(orig)

    assert deps["chat_store"].closed is False


# --- lifespan teardown resilience (#284) ---
#
# The teardown releases ten unrelated resources. As a bare statement chain, one
# step raising or hanging strands every resource after it -- the Qdrant client
# and five Redis pools then leak for the rest of the process' life. Each step
# is therefore individually bounded and individually guarded.
#
# The teardown step names below double as the assertion list, so they must stay
# in step with the teardown_steps tuple in app/main.py.
_TEARDOWN_STEPS = (
    "chat retention task",
    "chat store",
    "auth token purge task",
    "auth store",
    "qdrant client",
    "cache client",
    "analytics redis",
    "cost budget redis",
    "auth rate-limit redis",
    "readiness redis",
)

# The two background-task steps are the odd ones out: the lifespan cancels a
# task there rather than calling a close, so they are indexed into main.state
# instead of reached through an attribute.
_TASK_STEPS = {
    "chat retention task": "chat_retention",
    "auth token purge task": "auth_token_purge",
}

# Upper bound on how long a fixture that refuses to die is left running. It must
# be well under the test's own deadline: asyncio.run() cancels and gathers every
# pending task on the way out, so a task that ignores cancellation would stall
# teardown of the entire suite rather than fail one test.
_STUBBORN_MAX_SECONDS = 2.0


class _IgnoresCancellation:
    """A real asyncio.Task whose coroutine swallows CancelledError and keeps
    running: the in-process shape of a teardown step stuck on a socket that
    never answers.

    This is the one hang ``asyncio.wait_for`` does not save you from -- it
    waits for the cancellation to land before returning, so a task that refuses
    to unwind makes it block forever. Hence ``asyncio.wait`` in
    _cancel_and_wait. Self-terminating, so it can never outlive its test.
    """

    def __init__(self):
        self._task = None

    async def start(self):
        """Return the Task itself -- the one placed in main.state.

        A wrapper object would not do: ``_cancel_and_wait`` hands the value
        straight to ``asyncio.wait``, which needs a real Task/Future and
        raises AttributeError on anything else. That raise is swallowed by the
        step guard, so a wrapper silently turned the hang case into a copy of
        the raise case while still going green.
        """
        self._task = asyncio.get_running_loop().create_task(self._run())
        # Yielded to, so the coroutine is genuinely inside its sleep loop
        # before teardown cancels it. A task cancelled before its first step
        # never runs at all, which would make this fixture a no-op.
        await asyncio.sleep(0)
        return self._task

    async def _run(self):
        # Sliced sleeps, not one long one: the deadline has to be re-checked
        # often, because each cancel re-arms the sleep. A single 3600s sleep
        # would be re-entered after every CancelledError and the deadline
        # would never be reached -- leaving an immortal task that stalls
        # asyncio.run's own shutdown.
        deadline = asyncio.get_running_loop().time() + _STUBBORN_MAX_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    async def aclose(self):
        """Force it to stop, on the loop that created it (a task awaited from a
        second asyncio.run() belongs to a closed loop and raises)."""
        self._task.cancel()
        done, _pending = await asyncio.wait({self._task}, timeout=_STUBBORN_MAX_SECONDS + 5)
        assert self._task in done, "stubborn task outlived its own deadline"


async def _cancel_explodes():
    """A background task whose cancel() itself raises, killing the teardown step
    that owns it before it can await anything."""
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise RuntimeError("cancel exploded") from None


def _record_teardown(monkeypatch, deps, released, faults):
    """Make every lifespan teardown step observable, and fault the one named in
    ``faults`` (``{step_name: "raise" | "hang"}``).

    A step appends its name to ``released`` only after it has actually
    completed, so a name missing from the list means the resource was NOT
    released -- not merely that the step was entered. That distinction is what
    makes the ordering assertions below meaningful.
    """
    holdouts = []
    exploding = []

    async def _nothing():
        return None

    async def _run_step(name, close_it):
        if faults.get(name) == "hang":
            await asyncio.Event().wait()  # never set: this step never returns
        if faults.get(name) == "raise":
            raise RuntimeError(f"{name} exploded")
        await close_it()
        released.append(name)

    def _wrap_attribute(name, holder, attr="close"):
        original = getattr(holder, attr)

        async def close():
            await _run_step(name, original)

        monkeypatch.setattr(holder, attr, close)

    _wrap_attribute("chat store", deps["chat_store"])
    _wrap_attribute("auth store", deps["auth_store"])
    _wrap_attribute("qdrant client", deps["qdrant"])
    _wrap_attribute("cache client", deps["cache"])

    for owner, attr, name in (
        (main, "close_analytics", "analytics redis"),
        (main, "close_cost_budget", "cost budget redis"),
        (main.auth_module, "close_rate_redis", "auth rate-limit redis"),
        (main, "health_module_close_redis", "readiness redis"),
    ):

        async def closer(name=name):
            await _run_step(name, _nothing)

        monkeypatch.setattr(owner, attr, closer)

    # The stub's own background loops are swapped for observable ones. This runs
    # after startup because the lifespan reads main.state when it builds the
    # teardown step list.
    async def install_task_recorders():
        for name, state_key in _TASK_STEPS.items():
            original = main.state[state_key]
            original.cancel()
            await asyncio.gather(original, return_exceptions=True)

            if faults.get(name) == "hang":
                holdout = _IgnoresCancellation()
                replacement = await holdout.start()
                holdouts.append(holdout)
            elif faults.get(name) == "raise":
                # A real Task whose cancel() raises, for the same reason: the
                # value in main.state is handed to asyncio.wait. Retrieved
                # below so the raise does not surface as an unretrieved-task
                # error when the loop closes.
                replacement = asyncio.get_running_loop().create_task(_cancel_explodes())
                exploding.append(replacement)
                # Started, so teardown's cancel() actually reaches it: a task
                # cancelled before its first step never runs, which would make
                # this a clean release instead of the raise under test.
                await asyncio.sleep(0)
            else:

                async def _loop(name=name):
                    try:
                        await asyncio.sleep(3600)
                    finally:
                        released.append(name)

                replacement = asyncio.get_running_loop().create_task(_loop())
                # Let it actually reach its sleep: a task cancelled before its
                # first step never enters the coroutine body, so the `finally`
                # that records the release would never run.
                await asyncio.sleep(0)
            main.state[state_key] = replacement

    return install_task_recorders, holdouts, exploding


def test_lifespan_teardown_releases_every_resource_in_order(monkeypatch):
    """The complete, ordered release list on a clean shutdown.

    This is what makes the fault tests below trustworthy: a teardown step
    missing from the lifespan, or released out of order, fails here first -- so
    the subsets the fault tests assert are known to be real subsets of a
    working shutdown.
    """
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch)
    released = []

    async def scenario():
        async with main.lifespan(None):
            install, holdouts, _exploding = _record_teardown(monkeypatch, deps, released, {})
            await install()
        for holdout in holdouts:
            await holdout.aclose()

    try:
        _run(scenario())
    finally:
        _restore_state(orig)

    assert tuple(released) == _TEARDOWN_STEPS


@pytest.mark.parametrize("kind", ["raise", "hang"])
@pytest.mark.parametrize("faulty", _TEARDOWN_STEPS)
def test_lifespan_teardown_survives_a_broken_step(monkeypatch, caplog, faulty, kind):
    """One teardown step in turn is made to raise, then to hang. Every resource
    after it must still be released.

    The broken step is the only expected loss -- a close that raises is by
    definition not a close that released anything. What is under test is the
    tail: without per-step guarding, `cache client` (say) and the four Redis
    pools after it are skipped outright, and without a bound the hanging case
    never reaches them at all.
    """
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch)
    released = []
    # A short bound keeps the suite fast; it is the same code path as the 2s
    # production value and the hang is real either way.
    monkeypatch.setattr(main, "_TEARDOWN_CLOSE_TIMEOUT", 0.05)

    async def scenario():
        holdouts = []
        exploding = []
        try:
            async with main.lifespan(None):
                install, _h, exploding = _record_teardown(monkeypatch, deps, released, {faulty: kind})
                holdouts.extend(_h)
                await install()
        finally:
            # Same loop as the tasks' creator, and unconditional: leaving an
            # immortal pending task behind would stall asyncio.run()'s own
            # shutdown and hang the suite rather than fail this test.
            for holdout in holdouts:
                await holdout.aclose()
            # Retrieve the raises, so they are not reported a second time as
            # unretrieved task exceptions when the loop closes.
            for task in exploding:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*exploding, return_exceptions=True)

    # Ten steps at 0.05s cannot approach this. It exists so that a genuinely
    # unbounded step fails here instead of hanging the run.
    try:
        with caplog.at_level(logging.WARNING, logger="app.main"):
            _run(asyncio.wait_for(scenario(), timeout=30.0))
    finally:
        _restore_state(orig)

    # Every step except the broken one, in the original order: the steps before
    # it ran normally, and the ones after it must still run despite it.
    assert tuple(released) == tuple(step for step in _TEARDOWN_STEPS if step != faulty)
    # The operator can tell which resource leaked, without a debugger.
    assert faulty in caplog.text
    if kind == "hang" and faulty in _TASK_STEPS:
        # A hanging background task is abandoned by the inner asyncio.wait
        # budget inside _cancel_and_wait. Loosening that budget past the outer
        # one makes the outer wait_for fire first, so the step is reported as a
        # generic close timeout and this line never appears -- which is how the
        # halving stays pinned. Teardown would still reach the later steps;
        # what changes is that the refusal goes unreported.
        assert "ignored cancellation" in caplog.text
