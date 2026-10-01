"""Tests for the internal (non-HTTP) pipeline functions in app.main."""

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

# summary is omitted from the constructor here, unlike the shared default.
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
        self.encoded: list = []

    def encode(self, text):
        self.calls += 1
        self.encoded.append(text)
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
    """Records predict()'s exact pairs, so a test can pin which articles were rescored."""

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
    qdrant.scroll_pages = pages
    return qdrant


class _PagingQdrant:
    """Fake scroll serving a real point list one ``limit``-sized page per call."""

    def __init__(self, pts):
        self.pts = pts
        self.calls: list[dict] = []

    async def scroll(self, **kwargs):
        self.calls.append(kwargs)
        start = kwargs["offset"] or 0
        page = self.pts[start : start + kwargs["limit"]]
        nxt = start + len(page)
        return page, (nxt if nxt < len(self.pts) else None)

    def points_served(self) -> int:
        return sum(len(self.pts[c["offset"] or 0 : (c["offset"] or 0) + c["limit"]]) for c in self.calls)


def test_embed_sparse_returns_first_element():
    fake = _FakeSparse([1, 3], [0.9, 0.4])
    out = main._embed_sparse(fake, "query")
    assert fake.calls == 1
    assert out.indices.tolist() == [1, 3]
    assert out.values.tolist() == [0.9, 0.4]


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
    # recency stays in dt/ind (the ranking weight) and out of the retrieval query.
    assert (rq, fd, td) == ("deals", None, None)
    assert dt is None and ind is None


def test_effective_intent_normalizes_word_numbers():
    """'top ten ipo' must retrieve like 'top 10 ipo', not match titles like 'Ten Sports'."""
    assert main._effective_intent("top ten ipo", None, None) == ("top 10 ipo", None, None, None, None)
    assert main._effective_intent("top ten deals", None, None) == ("top 10 deals", None, None, None, None)
    assert main._effective_intent("top ten ipo", "2024-01-01", "2024-12-31") == (
        "top 10 ipo", "2024-01-01", "2024-12-31", None, None,
    )


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
    # both model names belong in the key, so swapping either invalidates it
    assert key.startswith("vec:")
    assert "dense-model" in key and "sparse-model" in key
    assert "query" in key
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


def _vector_key_setup(monkeypatch, fake_cache):
    cache = fake_cache()
    dense = _FakeDense([0.1, 0.2, 0.3])
    sparse = _FakeSparse([1, 3], [0.9, 0.4])
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    monkeypatch.setitem(main.state, "qdrant", _FakeQdrant(points=[]))
    monkeypatch.setattr(main.config, "EMBED_MODEL", "dense-model")
    monkeypatch.setattr(main.config, "SPARSE_MODEL", "sparse-model")
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    return cache, dense


def test_vector_cache_key_is_keyed_on_the_text_that_was_embedded(monkeypatch, fake_cache):
    """The key must be built from the same normalised text that is encoded, or a
    full-width query reuses an ASCII query's cached vectors."""
    cache, dense = _vector_key_setup(monkeypatch, fake_cache)

    _run(main.hybrid_search("ＴＥＳＴ deals", 8))

    assert dense.encoded == ["TEST deals"], "the encoder saw unnormalised text"
    key, _value, _ttl = cache.sets[0]
    assert "TEST deals" in key
    assert "ＴＥＳＴ" not in key


def test_vector_cache_key_is_bounded_for_a_long_query(monkeypatch, fake_cache):
    """The vector cache outlives the request under its own TTL, so an unbounded key
    is the costliest one to leave open."""
    cache, _dense = _vector_key_setup(monkeypatch, fake_cache)

    _run(main.hybrid_search("q" * 1000, 8))

    key, _value, _ttl = cache.sets[0]
    assert key.startswith("vec:")
    assert len(key) <= 256, f"vec key grew to {len(key)} chars: {key[:80]!r}"
    # the key holds a digest, not the 1000-char query text
    assert main._cache_key_component("q" * 1000) in key
    assert "q" * 200 not in key, "the raw query reached the key"


def test_vector_cache_key_carries_no_control_characters(monkeypatch, fake_cache):
    cache, _dense = _vector_key_setup(monkeypatch, fake_cache)

    _run(main.hybrid_search("test\x00\r\nINJECTED", 8))

    key, _value, _ttl = cache.sets[0]
    assert "\x00" not in key
    assert "\r" not in key and "\n" not in key


def test_vector_cache_key_changes_when_a_model_changes(monkeypatch, fake_cache):
    """A model swap must not reuse vectors encoded by the previous model."""
    cache, _dense = _vector_key_setup(monkeypatch, fake_cache)
    _run(main.hybrid_search("query", 8))
    first = cache.sets[0][0]

    monkeypatch.setattr(main.config, "EMBED_MODEL", "different-dense")
    _run(main.hybrid_search("query", 8))

    assert cache.sets[1][0] != first


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
    """Uses the real writer's payload, so a break on either the write or the read
    side fails here; an empty stored value reads back as None."""
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


def _date_window_qdrant(monkeypatch):
    """Two in-window articles newest-first, as Qdrant's published_date order_by returns them."""
    page = (
        [
            _Point(11, {"title": "Newer", "published_date": "2025-06-02T00:00:00"}),
            _Point(12, {"title": "Older", "published_date": "2025-05-02T00:00:00"}),
        ],
        None,
    )
    # two pages: the knob tests retrieve twice and the fake serves one page per call
    qdrant = _FakeQdrant(scroll_pages=[page, page])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    return qdrant


def _date_window_articles():
    return _run(main.retrieve_by_date_window(top_k=5, from_date="2025-05-01", to_date="2025-06-30"))


def test_date_fillers_take_their_own_knob(monkeypatch):
    """Pinned to the shipped 0.2 first so a developer .env cannot decide the starting expectation."""
    _date_window_qdrant(monkeypatch)
    monkeypatch.setattr(main.config, "DATE_FILLER_SCORE", 0.2)
    assert [a.score for a in _date_window_articles()] == [0.2, 0.2]

    monkeypatch.setattr(main.config, "DATE_FILLER_SCORE", 0.05)
    assert [a.score for a in _date_window_articles()] == [0.05, 0.05]


def test_date_fillers_do_not_follow_the_inclusion_gate_at_the_call_site(monkeypatch):
    """The floor must not be re-derived from ASK_MIN_SCORE where the articles are built."""
    _date_window_qdrant(monkeypatch)
    monkeypatch.setattr(main.config, "DATE_FILLER_SCORE", 0.2)
    monkeypatch.setattr(main.config, "ASK_MIN_SCORE", 0.9)
    assert [a.score for a in _date_window_articles()] == [0.2, 0.2]


def test_date_filler_knob_ships_the_value_the_alias_resolved_to(parse_config):
    """chat drops sources scoring below ASK_MIN_SCORE, so a filler floor under the gate
    would be dropped before the model saw it."""
    shipped = parse_config()
    assert shipped.ASK_MIN_SCORE == 0.2
    assert shipped.DATE_FILLER_SCORE == 0.2
    assert shipped.DATE_FILLER_SCORE >= shipped.ASK_MIN_SCORE


def test_retuning_the_inclusion_gate_does_not_move_the_date_filler_floor(parse_config):
    parsed = parse_config(ASK_MIN_SCORE="0.9")
    assert parsed.ASK_MIN_SCORE == 0.9
    assert parsed.DATE_FILLER_SCORE == 0.2


def test_date_filler_floor_is_tunable_from_the_environment(parse_config):
    assert parse_config(DATE_FILLER_SCORE="0.45").DATE_FILLER_SCORE == 0.45


def _filler_gate_warnings(caplog):
    return [r for r in caplog.records if "DATE_FILLER_SCORE" in r.getMessage()]


def test_a_filler_floor_below_the_chat_gate_is_logged_at_startup(parse_config, caplog):
    """A mis-ordered pairing fails silently -- chat drops every filler -- so config.py
    warns rather than raising and the service still serves."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config(ASK_MIN_SCORE="0.6")
    warnings = _filler_gate_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "0.2" in message and "0.6" in message
    assert "temporal fallback" in message


def test_the_filler_floor_warning_stays_quiet_for_sound_configurations(parse_config, caplog):
    """A warning that fires for sound configs trains the reader to ignore the one that matters."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config()
    assert _filler_gate_warnings(caplog) == []

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config(ASK_MIN_SCORE="0.6", DATE_FILLER_SCORE="0.7")
    assert _filler_gate_warnings(caplog) == []

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        parse_config(ASK_MIN_SCORE="0.6", DATE_FILLER_SCORE="0.6")
    assert _filler_gate_warnings(caplog) == []  # equal is fine: the gate is `>=`


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
    """With the gate off nothing may reach the reranker, or a caller that forgets
    the check silently pays the cross-encoder pass again."""
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
    """Every prediction is paid under the global inference lock, so a wide chat
    shortlist must not mean a prediction per article."""
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
    """The limited budget goes to the highest body-window overlap -- not to the top
    scorer, and not to the first articles in list order."""
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
    # Qdrant may return a string point id; it must still match the int-id article
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
    # index 2 skips the meta line and the summary; the note marks the cut so the
    # model can tell a capped excerpt from a short article
    excerpt = out.split("\n", 2)[2]
    assert excerpt == "x" * 20 + main.BODY_TRUNCATION_NOTE


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


def test_facet_values_match_a_full_scan_of_a_multi_page_collection(monkeypatch):
    """The page size may change the round trips, never the values -- including a
    value that exists only at the very end of the collection."""
    points = [_Point(i, {"industry_names": [f"Industry {i % 40}"]}) for i in range(5000)]
    points.append(_Point(5000, {"industry_names": ["Tail Only"]}))

    qdrant = _PagingQdrant(points)
    monkeypatch.setitem(main.state, "qdrant", qdrant)

    out = _run(main._facet_values("industry_names"))

    reference = sorted({p.payload["industry_names"][0] for p in points})
    assert out == reference
    assert "Tail Only" in out
    assert len(qdrant.calls) == math.ceil(len(points) / main.FACET_SCROLL_PAGE)
    assert all(c["limit"] == main.FACET_SCROLL_PAGE for c in qdrant.calls)
    assert qdrant.points_served() == len(points)
    round_trips = len(qdrant.calls)

    old = _PagingQdrant(points)
    monkeypatch.setitem(main.state, "qdrant", old)
    monkeypatch.setattr(main, "FACET_SCROLL_PAGE", 256)

    assert _run(main._facet_values("industry_names")) == out
    assert len(old.calls) > round_trips
    assert old.points_served() == len(points)


def test_facet_values_are_the_same_at_both_page_sizes_when_the_cap_trips(monkeypatch):
    """The cap is checked every FACET_CAP_CHECK_EVERY points CONSUMED, not once per
    page, so a truncated walk stops at the same scroll position at either size."""
    points = [_Point(i, {"industry_names": [f"V{(i * 7) % 1024:04d}"]}) for i in range(1024)]

    # every value is distinct, so the cap trips inside the first check interval
    consumed = main.FACET_CAP_CHECK_EVERY
    reference = sorted({p.payload["industry_names"][0] for p in points[:consumed]})[: main.FACETS_LIMIT]
    assert len(reference) == main.FACETS_LIMIT

    qdrant = _PagingQdrant(points)
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    at_shipped = _run(main._facet_values("industry_names"))

    assert at_shipped == reference

    old = _PagingQdrant(points)
    monkeypatch.setitem(main.state, "qdrant", old)
    monkeypatch.setattr(main, "FACET_SCROLL_PAGE", 256)

    assert _run(main._facet_values("industry_names")) == reference
    # the shipped page size reads the same points in one round trip
    assert len(qdrant.calls) == 1
    assert len(old.calls) == consumed // 256


def test_best_body_window_tail_wins_on_tie():
    body = ("filler " * 60) + "alpha beta gamma"
    tokens = {"alpha", "beta", "gamma"}
    out = main._best_body_window(body, tokens, 50, 50)
    assert out == body[-50:]


def test_effective_step_leaves_the_default_scan_alone():
    positions = 50_000 - 1500 + 1
    assert main._effective_step(positions, 500, 200) == 500
    assert -(-positions // main._effective_step(positions, 500, 200)) == 98


def test_effective_step_widens_a_legal_but_expensive_stride():
    """Clamping the step alone leaves 48,501 windows of a 50K body to score, and the
    candidate cap applies only after every body-bearing article is scanned."""
    positions = 49_700 - 1500 + 1
    widened = main._effective_step(positions, 1, 200)
    assert widened > 1
    assert -(-positions // widened) <= 200


def test_effective_step_never_returns_a_zero_stride():
    """`range(0, n, 0)` raises ValueError, and `_best_body_window` is module-level
    and directly callable, so it cannot rely on every caller going via config."""
    widened = main._effective_step(1000, 0, 200)
    assert widened >= 1
    assert -(-1000 // widened) <= 200
    # A non-positive budget disables widening rather than producing stride 0.
    assert main._effective_step(1000, 0, 0) == 1
    out = main._best_body_window("filler " * 3000, {"alpha"}, 1500, 0)
    assert isinstance(out, str) and out


def test_best_body_window_finds_dense_region_at_the_default_budget():
    body = ("filler " * 4000) + ("alpha beta gamma " * 40) + ("filler " * 2000)
    out = main._best_body_window(body, {"alpha", "beta", "gamma"}, 1500, 500, max_windows=200)
    low = out.lower()
    assert "alpha" in low and "gamma" in low


def test_best_body_window_a_tight_budget_can_straddle_the_dense_region():
    """max_windows=1 forces the single window at start=0, so the result is
    deterministically body[:1500] -- the cost of a budget tighter than the stride."""
    body = ("filler " * 4000) + ("alpha beta gamma " * 40) + ("filler " * 2000)
    out = main._best_body_window(body, {"alpha", "beta", "gamma"}, 1500, 1, max_windows=1)
    assert out == body[:1500]
    unbudgeted = main._best_body_window(body, {"alpha", "beta", "gamma"}, 1500, 1)
    assert "alpha" in unbudgeted.lower()


class _IterationCountingTokens(set):
    """`_best_body_window` scores each window by iterating the token set, so the
    iteration count is the number of windows scored."""

    def __init__(self, items):
        super().__init__(items)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()

    def windows_scored(self):
        return self.iterations - 1


def test_best_body_window_applies_the_window_budget():
    """The budget must be enforced by `_best_body_window`, not merely available on
    `_effective_step`: deleting the line that applies it keeps the suite green."""
    body = "filler " * 7100  # 49,700 chars
    tokens = _IterationCountingTokens({"alpha", "beta", "gamma"})
    main._best_body_window(body, tokens, 1500, 1, max_windows=200)
    # Unbudgeted, step=1 would score 48,201 windows.
    assert tokens.windows_scored() <= 200, tokens.windows_scored()


def test_best_body_window_does_not_widen_a_default_scan():
    body = "filler " * 7100
    tokens = _IterationCountingTokens({"alpha", "beta", "gamma"})
    main._best_body_window(body, tokens, 1500, 500, max_windows=200)
    assert tokens.windows_scored() == 97  # ceil(48201 / 500)


def _stub_lifespan_deps(monkeypatch, chat_connect_error=None, auth_connect_error=None):
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

    reported = []

    async def _report_legacy_password_hashes():
        reported.append(True)
        return 0

    monkeypatch.setattr(main.auth_module, "report_legacy_password_hashes", _report_legacy_password_hashes)

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
        "reported": reported,
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
            assert deps["reported"] == [True]

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
    """A bad key must be named at startup rather than surfacing as a per-turn 401,
    and the process must stay up to serve /health."""
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
    """An ERROR line on every healthy boot trains operators to ignore the one that matters."""
    orig = dict(main.state)
    # assembled rather than written out: a credential-shaped literal next to
    # GEMINI_API_KEY is what a secrets scanner flags
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


# must stay in step with the teardown_steps tuple in app/main.py
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

# these two cancel a task in main.state rather than calling a close, so they are
# looked up by state key instead of by attribute
_TASK_STEPS = {
    "chat retention task": "chat_retention",
    "auth token purge task": "auth_token_purge",
}

# far below any test deadline: asyncio.run() gathers every pending task on the way
# out, so a task ignoring cancellation would stall the whole suite, not one test
_STUBBORN_MAX_SECONDS = 2.0


class _IgnoresCancellation:
    """A real Task whose coroutine swallows CancelledError -- the one hang
    ``asyncio.wait_for`` cannot escape, so it terminates on its own deadline."""

    def __init__(self):
        self._task = None

    async def start(self):
        """Returns the real Task: asyncio.wait rejects a wrapper, and the step guard
        would swallow that error, turning the hang case into the raise case."""
        self._task = asyncio.get_running_loop().create_task(self._run())
        # so the task is inside its sleep before cancel: one cancelled before its
        # first step never runs
        await asyncio.sleep(0)
        return self._task

    async def _run(self):
        # sliced sleeps: a cancel re-arms a single long sleep, so the deadline
        # would never be reached and the task would be immortal
        deadline = asyncio.get_running_loop().time() + _STUBBORN_MAX_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue

    async def aclose(self):
        """Stops it on the loop that created it; awaited from a second asyncio.run()
        it belongs to a closed loop and raises."""
        self._task.cancel()
        done, _pending = await asyncio.wait({self._task}, timeout=_STUBBORN_MAX_SECONDS + 5)
        assert self._task in done, "stubborn task outlived its own deadline"


async def _cancel_explodes():
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise RuntimeError("cancel exploded") from None


def _record_teardown(monkeypatch, deps, released, faults):
    """A step appends its name to ``released`` only after it completes, so a name
    missing from the list means the resource was NOT released, not merely entered."""
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

    # runs after startup: the lifespan reads main.state when it builds the step list
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
                # a real Task whose cancel() raises, for the same reason: the value in
                # main.state is handed to asyncio.wait
                replacement = asyncio.get_running_loop().create_task(_cancel_explodes())
                exploding.append(replacement)
                # started so teardown's cancel() reaches it: one cancelled before its
                # first step never runs, making this a clean release
                await asyncio.sleep(0)
            else:

                async def _loop(name=name):
                    try:
                        await asyncio.sleep(3600)
                    finally:
                        released.append(name)

                replacement = asyncio.get_running_loop().create_task(_loop())
                # so the `finally` that records the release can run at all: a task
                # cancelled before its first step never enters the body
                await asyncio.sleep(0)
            main.state[state_key] = replacement

    return install_task_recorders, holdouts, exploding


def test_lifespan_teardown_releases_every_resource_in_order(monkeypatch):
    """The complete, ordered release list on a clean shutdown: a step missing from
    the lifespan, or released out of order, fails here before the fault tests."""
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
    """Without per-step guarding the later steps are skipped outright, and without
    a bound the hanging case never reaches them at all."""
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch)
    released = []
    # A short bound keeps the suite fast; same code path as the production value.
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
            # the tasks' creator loop, and unconditional: an immortal pending task
            # would stall asyncio.run()'s own shutdown
            for holdout in holdouts:
                await holdout.aclose()
            # retrieved so they are not re-reported as unretrieved when the loop closes
            for task in exploding:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*exploding, return_exceptions=True)

    # ten steps at 0.05s cannot approach this; an unbounded step must fail, not hang
    try:
        with caplog.at_level(logging.WARNING, logger="app.main"):
            _run(asyncio.wait_for(scenario(), timeout=30.0))
    finally:
        _restore_state(orig)

    assert tuple(released) == tuple(step for step in _TEARDOWN_STEPS if step != faulty)
    assert faulty in caplog.text
    if kind == "hang" and faulty in _TASK_STEPS:
        # the inner asyncio.wait budget in _cancel_and_wait must stay below the outer
        # one, or the step reads as a generic close timeout instead
        assert "ignored cancellation" in caplog.text
