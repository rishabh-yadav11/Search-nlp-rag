"""Regression tests for issue #252: input normalisation, facet bounds and
unambiguous cache keys.

Each test drives the real ``/search`` endpoint or the real helpers, and asserts
on observable behaviour (response body, cache state, filter contents) rather
than on the internals of the normalisation helpers. The two bugs that returned
the *wrong* answer -- two different facet sets sharing one cache key, and a
facet bomb building a multi-kilobyte filter -- are pinned end to end.
"""
import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client.models import MatchAny

from app import auth, main
from app.config import config
from app.input_hygiene import (
    MAX_FACET_VALUE_LEN,
    MAX_FACET_VALUES,
    MAX_KEY_LEN,
    build_cache_key,
    normalize_text,
    split_facet_values,
)

_client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _rate_limiter(monkeypatch):
    """The public endpoints fail closed (503) when the limiter Redis is down, so
    these tests get a counting stub instead. Rebuilt per test, no counter leaks."""

    class _FakeRateRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            return 1

    monkeypatch.setattr(auth, "_rate_client", _FakeRateRedis())
    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 10_000)


class _RecordingCache:
    """In-memory cache that remembers every key it was asked for, so a test can
    tell a genuine cache miss from a hit on another request's entry."""

    def __init__(self):
        self.store: dict = {}
        self.gets: list = []
        self.sets: list = []

    async def get(self, key):
        self.gets.append(key)
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append(key)


def _summary(id_, score=0.5, industry="Fintech"):
    return {
        "id": id_, "title": f"Title {id_}", "url": f"https://example.com/{id_}",
        "published_date": "2025-01-10", "category": "News",
        "summary": f"summary {id_}", "score": score,
        "author_names": ["A"], "industry_names": [industry],
        "dealtype_names": ["Funding"],
    }


def _wire(monkeypatch, *, results_for=None, pass_dates=False):
    """Point /search at a recording cache and a retrieval stub.

    ``results_for`` maps the (industry, dealtype) pair a retrieval is asked for
    to the article ids it returns, so a result set that does not match the
    request's own facets is visible as a wrong answer rather than hidden behind
    a single canned response. ``pass_dates`` keeps from_date/to_date instead of
    dropping them, which the date-validation tests need in order to reach the
    filter builder.
    """
    cache = _RecordingCache()
    retrieved: list = []
    recorded: list = []

    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, ""))
    if pass_dates:
        monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, fd, td, None, None))
    else:
        monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "suggested_top_k", lambda q: None)
    monkeypatch.setattr(config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(config, "ENABLE_DIVERSITY", False)

    async def _record_search(query, *args, **kwargs):
        recorded.append(query)

    monkeypatch.setattr(main, "record_search", _record_search)

    async def _retrieve(retrieval_q, top_k, **kwargs):
        facets = (kwargs.get("industry"), kwargs.get("dealtype"))
        retrieved.append(facets)
        ids = results_for(facets) if results_for else [1]
        articles = [main.SourceArticle.model_validate(_summary(i)) for i in ids]
        return articles, facets[0], facets[1], kwargs.get("content_type")

    monkeypatch.setattr(main, "retrieve_with_auto_facet_fallback", _retrieve)
    return cache, retrieved, recorded


# --- the ambiguous cache key ------------------------------------------------

# These two facet sets built the *same* cache token when the six fields were
# joined with '|', because a value could itself contain the delimiter:
# 'a' + '|' + 'b|c' and 'a|b' + '|' + 'c' are the same string. They select
# genuinely different filters, so the second request was served the first
# request's results.
_COLLIDING_A = {"industry": "a", "dealtype": "b|c"}
_COLLIDING_B = {"industry": "a|b", "dealtype": "c"}


def test_colliding_facet_sets_get_distinct_cache_entries(monkeypatch):
    cache, retrieved, _ = _wire(
        monkeypatch, results_for=lambda f: [1] if f == ("a", "b|c") else [99]
    )

    first = _client.get("/search", params={"q": "test", **_COLLIDING_A})
    second = _client.get("/search", params={"q": "test", **_COLLIDING_B})

    assert first.status_code == 200
    assert second.status_code == 200
    # Two different filters, so retrieval must run twice -- the second request
    # must not be answered from the first request's cache entry.
    assert retrieved == [("a", "b|c"), ("a|b", "c")]
    assert second.json()["cached"] is False
    assert len(cache.gets) == 2
    assert cache.gets[0] != cache.gets[1], "the two facet sets share one cache key"
    assert len(cache.store) == 2, "one cache entry was written for two different filters"


def test_colliding_facet_sets_get_distinct_results(monkeypatch):
    """The correctness half of the same bug: the wrong result set was served."""
    _wire(monkeypatch, results_for=lambda f: [1] if f == ("a", "b|c") else [99])

    first = _client.get("/search", params={"q": "test", **_COLLIDING_A})
    second = _client.get("/search", params={"q": "test", **_COLLIDING_B})

    assert [r["id"] for r in first.json()["results"]] == [1]
    assert [r["id"] for r in second.json()["results"]] == [99], \
        "request 2 was served request 1's results for a filter it never ran"


def test_facet_cache_token_is_injective_over_delimiter_confusion():
    """The property that fixes the collision: no two distinct field tuples
    produce one token, whatever the values contain."""
    token_a = main.facet_cache_token("a", "b|c", None, None, None, None)
    token_b = main.facet_cache_token("a|b", "c", None, None, None, None)
    assert token_a != token_b


@pytest.mark.parametrize("left,right", [
    # Each pair shifts a '|' across a field boundary. A delimiter-joined
    # encoding renders both as the same string; a length-prefixed one cannot.
    (("a", "b|c"), ("a|b", "c")),
    (("a", "b|c|d"), ("a|b", "c|d")),
    (("a|b", "c"), ("a", "b|c")),
    (("|", "a"), ("", "|a")),
    (("a|", "b"), ("a", "|b")),
    (("ab", "c"), ("a", "bc")),
])
def test_facet_cache_token_never_collides_across_field_boundaries(left, right):
    assert left != right, "the pair under test must be genuinely different"
    token_left = main.facet_cache_token(*left, None, None, None, None)
    token_right = main.facet_cache_token(*right, None, None, None, None)
    assert token_left != token_right
    # And they really do select different filters, so the key has to differ.
    assert main.build_facet_filter(*left, None, None, None, None) != \
        main.build_facet_filter(*right, None, None, None, None)


def test_build_cache_key_is_injective():
    """A length-prefixed encoding is unambiguous: shifting content between
    parts cannot produce the same key."""
    assert build_cache_key("a", "b|c") != build_cache_key("a|b", "c")
    assert build_cache_key("ab", "c") != build_cache_key("a", "bc")
    assert build_cache_key("a", "b") != build_cache_key("a", "b", "")


def test_build_cache_key_bounds_long_keys():
    """A pathological input is digested, so one request cannot create an
    arbitrarily long key."""
    key = build_cache_key("q" * 100_000, namespace="retrieve")
    assert len(key) <= len("retrieve:sha256:") + 64
    assert key.startswith("retrieve:sha256:")


def test_build_cache_key_keeps_normal_keys_readable_and_namespaced():
    key = build_cache_key("fintech funding", 8, "", namespace="retrieve")
    assert key.startswith("retrieve:")
    assert "fintech funding" in key
    assert len(key) <= MAX_KEY_LEN


# --- facet bounds -----------------------------------------------------------


def test_facet_value_count_over_the_cap_is_rejected():
    too_many = ",".join(f"v{i}" for i in range(MAX_FACET_VALUES + 1))
    with pytest.raises(HTTPException) as excinfo:
        split_facet_values("industry", too_many)
    assert excinfo.value.status_code == 400
    assert "industry" in excinfo.value.detail
    assert str(MAX_FACET_VALUES) in excinfo.value.detail
    # The rejected values are caller input and must not be reflected back.
    assert "v0" not in excinfo.value.detail


def test_facet_value_length_over_the_cap_is_rejected():
    with pytest.raises(HTTPException) as excinfo:
        split_facet_values("author", "n" * (MAX_FACET_VALUE_LEN + 1))
    assert excinfo.value.status_code == 400
    assert "author" in excinfo.value.detail
    assert "n" * 20 not in excinfo.value.detail


def test_facet_bomb_is_rejected_rather_than_truncated():
    """A 5000-value facet must be refused outright: silently keeping the first
    ten would answer a different question than the one asked."""
    bomb = ",".join(f"v{i}" for i in range(5000))
    with pytest.raises(HTTPException) as excinfo:
        main.build_facet_filter(bomb, None, None, None, None, None)
    assert excinfo.value.status_code == 400
    # Refused, not trimmed: an error, so the caller learns the facet was dropped
    # rather than receiving results for the first ten values only.
    assert "industry" in excinfo.value.detail


def test_facet_filter_never_builds_an_oversized_match_any():
    """The cap has to hold at the filter itself, not only at the HTTP edge --
    the chat and date-window paths reach build_facet_filter directly."""
    with pytest.raises(HTTPException):
        main.build_facet_filter("y" * 5000, None, None, None, None, None)


def test_facets_at_the_cap_are_still_accepted():
    """The cap must not reject a legitimate in-bounds request."""
    values = ["v" * MAX_FACET_VALUE_LEN] * MAX_FACET_VALUES
    accepted = split_facet_values("industry", ",".join(values))
    assert len(accepted) == MAX_FACET_VALUES
    assert all(len(v) == MAX_FACET_VALUE_LEN for v in accepted)
    filt = main.build_facet_filter(",".join(values), None, None, None, None, None)
    assert len(filt.must[0].match.any) == MAX_FACET_VALUES


def test_oversized_facet_over_http_is_a_400_not_a_200(monkeypatch):
    _wire(monkeypatch)
    bomb = ",".join(f"v{i}" for i in range(5000))
    response = _client.get("/search", params={"q": "test", "industry": bomb})
    assert response.status_code == 400
    assert "industry" in response.json()["detail"]


def test_oversized_facet_never_reaches_a_cache_key(monkeypatch):
    """The /search cache key is built before the retrieval pipeline runs, so
    the bound has to be applied before the key, not only inside the filter."""
    cache, _, _ = _wire(monkeypatch)
    _client.get("/search", params={"q": "test", "industry": ",".join(f"v{i}" for i in range(5000))})
    assert cache.gets == [] and cache.store == {}


def test_facet_values_are_normalised_into_the_filter(monkeypatch):
    """Full-width and control characters in a facet value must not reach Qdrant."""
    filt = main.build_facet_filter("Fintech\x00, Ｈealthtech", None, None, None, None, None)
    assert filt.must[0].match.any == ["Fintech", "Healthtech"]


# --- static error messages --------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("from_date", "not-a-date"),
    ("to_date", "not-a-date"),
    ("from_date", "<script>alert(1)</script>"),
    ("from_date", "A" * 5000),
])
def test_invalid_date_returns_a_static_message_that_does_not_echo(field, value):
    kwargs = {"industry": None, "dealtype": None, "author": None, "from_date": None, "to_date": None}
    kwargs[field] = value
    with pytest.raises(HTTPException) as excinfo:
        main.build_facet_filter(**kwargs)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == f"invalid {field}"
    # None of the caller's input may appear anywhere in the message.
    assert value not in excinfo.value.detail
    assert "script" not in excinfo.value.detail
    assert "not-a-date" not in excinfo.value.detail


def test_invalid_date_error_body_is_bounded(monkeypatch):
    """End to end: a long invalid date must produce a small static 400 body.

    The real fallback path runs here (only the retrieval itself is stubbed) so
    the request actually reaches the filter builder that validates the date.
    """
    cache = _RecordingCache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, ""))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, fd, td, None, None))
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "suggested_top_k", lambda q: None)

    async def _record_search(*args, **kwargs):
        return None

    async def _retrieve_and_rerank(*args, **kwargs):
        return []

    monkeypatch.setattr(main, "record_search", _record_search)
    monkeypatch.setattr(main, "retrieve_and_rerank", _retrieve_and_rerank)

    response = _client.get("/search", params={"q": "test", "from_date": "Z" * 5000})
    assert response.status_code == 400
    assert len(response.text) < 200
    # Nothing is ever written for an invalid date, so no entry can exist to be
    # replayed later as a valid answer for this key.
    assert cache.sets == []


# --- query normalisation ----------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("ＴＥＳＴ", "TEST"),                      # full-width Latin
    ("ﬁntech", "fintech"),                    # fi ligature
    ("café", "café"),                    # NFD composed by NFKC
    ("test\x00injected", "testinjected"),      # NUL removed
    ("test\r\nINJECTED", "testINJECTED"),      # CRLF removed
    ("test \t\n q", "test q"),                 # whitespace runs collapse
    ("test 　 q", "test q"),           # ideographic space
    ("  padded  ", "padded"),                 # stripped
    ("\x7fDEL", "DEL"),                        # DEL removed
])
def test_normalize_text_canonicalises(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_preserves_case():
    """Case reaches the embedder, so it is deliberately not folded: folding it
    would change what a query retrieves, not just how it is keyed."""
    assert normalize_text("Test IPO") == "Test IPO"


def test_normalize_text_leaves_no_control_characters():
    raw = "a\x00b\x01c\x1fd\x7fe"
    cleaned = normalize_text(raw)
    assert not any(ch < " " or ch == "\x7f" for ch in cleaned)


def test_equivalent_spellings_share_one_cache_entry(monkeypatch):
    """Full-width and ASCII spellings are the same question, so they must
    resolve to one cache entry rather than each embedding and retrieving."""
    cache, retrieved, _ = _wire(monkeypatch)

    ascii_response = _client.get("/search", params={"q": "fintech deals"})
    fullwidth_response = _client.get("/search", params={"q": "fｉｎｔｅｃｈ deals"})

    assert ascii_response.status_code == 200
    assert fullwidth_response.json()["cached"] is True
    assert len(retrieved) == 1, "the variant spelling re-ran the whole pipeline"
    assert len(cache.store) == 1
    # Same entry means the same answer, which is what makes sharing safe.
    assert fullwidth_response.json()["results"] == ascii_response.json()["results"]


def test_ligature_and_composed_spellings_share_one_cache_entry(monkeypatch):
    cache, retrieved, _ = _wire(monkeypatch)

    _client.get("/search", params={"q": "fintech office"})
    ligature = _client.get("/search", params={"q": "ﬁntech oﬃce"})

    assert ligature.json()["cached"] is True
    assert len(retrieved) == 1
    assert len(cache.store) == 1


def test_query_control_characters_reach_no_sink(monkeypatch):
    """A NUL/CRLF-bearing query must not survive into any durable record: the
    cache key, the analytics record or the echoed response. Those are the sinks
    a newline in a key or an analytics row would corrupt."""
    cache, _, recorded = _wire(monkeypatch)

    response = _client.get("/search", params={"q": "test\x00\r\nINJECTED: admin"})

    assert response.status_code == 200
    keys = cache.gets + cache.sets
    assert keys, "expected the request to build a cache key"
    for key in keys:
        assert "\x00" not in key
        assert "\r" not in key and "\n" not in key
    for query in recorded:
        assert "\x00" not in query
        assert "\r" not in query and "\n" not in query
    assert "\x00" not in response.json()["query"]
    assert "\n" not in response.json()["query"]


def test_control_characters_do_not_forge_a_second_cache_entry(monkeypatch):
    """Two requests differing only by an embedded newline must not collapse
    into one key, and neither may inject a line break into a key."""
    cache, _, _ = _wire(monkeypatch)

    _client.get("/search", params={"q": "test one"})
    _client.get("/search", params={"q": "test two\nforged line"})

    keys = cache.sets
    assert len(cache.store) == 2, "distinct queries collapsed into one cache entry"
    for key in keys:
        assert "\n" not in key


def test_query_of_only_control_characters_is_rejected(monkeypatch):
    """min_length=1 lets a string of control characters through, and those
    normalise away to nothing -- retrieving for an empty query is not an answer."""
    _wire(monkeypatch)
    response = _client.get("/search", params={"q": "\x00\x01\r\n"})
    assert response.status_code == 400
    assert response.json()["detail"] == "empty query"


def test_search_cache_key_is_bounded(monkeypatch):
    """The /search key is a second, independent key a query reaches. It has to
    be bounded on its own terms -- covering only the retrieve key would leave a
    long query to build an arbitrarily long /search key."""
    cache, _, _ = _wire(monkeypatch)
    response = _client.get("/search", params={"q": "q" * 1000})
    assert response.status_code == 200
    assert cache.gets, "expected the request to build a cache key"
    for key in cache.gets:
        assert key.startswith("search:sha256:"), "an over-long query must be digested"
        assert len(key) == len("search:sha256:") + 64


def test_search_cache_key_stays_bounded_with_a_max_sized_facet(monkeypatch):
    """A long-but-in-bounds facet set must not push the key past the bound
    either, since the token is part of the same key. This one stays under the
    limit and must remain readable rather than being digested for nothing."""
    cache, _, _ = _wire(monkeypatch)
    values = ",".join("v" * MAX_FACET_VALUE_LEN for _ in range(MAX_FACET_VALUES))
    response = _client.get("/search", params={"q": "test", "industry": values})
    assert response.status_code == 200
    for key in cache.gets:
        assert key.startswith("search:")
        assert len(key) <= MAX_KEY_LEN


def test_retrieve_cache_key_is_bounded_and_control_free(monkeypatch):
    """The retrieve-level key is the other key a query reaches (chat shares
    this pipeline), so it has to be bounded and cleaned on the same terms."""
    cache = _RecordingCache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, ""))
    monkeypatch.setattr(main, "_retrieval_queries", lambda q: [])

    async def _rerank(*args, **kwargs):
        return [main.SourceArticle.model_validate(_summary(1))]

    monkeypatch.setattr(main, "rerank", _rerank)

    asyncio.run(main.retrieve_and_rerank("q" * 100_000, 8, None))
    assert cache.sets, "expected a retrieve cache write"
    for key in cache.sets:
        assert key.startswith("retrieve:")
        assert len(key) <= len("retrieve:sha256:") + 64
        assert "\x00" not in key and "\n" not in key


def test_normalised_filter_values_reach_qdrant_clean(monkeypatch):
    """The retrieve key is built from _filter_token, so a facet carrying a NUL
    would otherwise put one straight into that key."""
    filt = main.build_facet_filter("Fintech\x00", None, None, None, None, None)
    token = main._filter_token(filt)
    assert "\x00" not in token
    assert "\n" not in token
    assert isinstance(filt.must[0].match, MatchAny)
