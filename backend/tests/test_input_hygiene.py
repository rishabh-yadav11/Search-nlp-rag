"""Input normalisation, facet bounds and unambiguous cache keys.

Each test drives the real ``/search`` endpoint or the real helpers, and asserts
on observable behaviour (response body, cache state, filter contents) rather
than on the internals of the normalisation helpers.
"""
import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client.models import MatchAny

from app import analytics, auth, main
from app.config import config
from app.input_hygiene import (
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

    async def get_many(self, keys):
        """Positional MGET: a batched read is still a read of every key, so it is
        recorded in ``gets`` exactly as separate ``get`` calls would be."""
        return [await self.get(key) for key in keys]

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
    """``results_for`` maps the (industry, dealtype) pair a retrieval is asked for
    to the article ids it returns, so a result set that does not match the
    request's own facets is visible as a wrong answer rather than hidden behind
    a single canned response. ``pass_dates`` keeps from_date/to_date so the
    date-validation tests can reach the filter builder."""
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

# These two facet sets built the *same* cache token when the fields were joined
# with '|': 'a' + '|' + 'b|c' and 'a|b' + '|' + 'c' are the same string. They
# select different filters, so the second request was served the first one's
# results.
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
    # Two different filters, so retrieval must run twice.
    assert retrieved == [("a", "b|c"), ("a|b", "c")]
    assert second.json()["cached"] is False
    # /search reads its own summary entry and the underlying retrieval entry in
    # one MGET, so keys are compared per namespace: the property is that the two
    # colliding facet sets never land on one key, not how many reads took.
    search_keys = [k for k in cache.gets if k.startswith("search:")]
    retrieve_keys = [k for k in cache.gets if k.startswith("retrieve:")]
    assert len(set(search_keys)) == 2, "the two facet sets share one search key"
    assert len(set(retrieve_keys)) == 2, "the two facet sets share one retrieve key"
    stored_search = [k for k in cache.store if k.startswith("search:")]
    assert len(stored_search) == 2, "one cache entry was written for two different filters"


def test_colliding_facet_sets_get_distinct_results(monkeypatch):
    _wire(monkeypatch, results_for=lambda f: [1] if f == ("a", "b|c") else [99])

    first = _client.get("/search", params={"q": "test", **_COLLIDING_A})
    second = _client.get("/search", params={"q": "test", **_COLLIDING_B})

    assert [r["id"] for r in first.json()["results"]] == [1]
    assert [r["id"] for r in second.json()["results"]] == [99], \
        "request 2 was served request 1's results for a filter it never ran"


def test_facet_cache_token_is_injective_over_delimiter_confusion():
    """No two distinct field tuples produce one token, whatever the values
    contain."""
    token_a = main.facet_cache_token("a", "b|c", None, None, None, None)
    token_b = main.facet_cache_token("a|b", "c", None, None, None, None)
    assert token_a != token_b


@pytest.mark.parametrize("left,right", [
    # Each pair shifts a '|' across a field boundary: a delimiter-joined encoding
    # renders both as the same string, a length-prefixed one cannot.
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
    # They really do select different filters, so the key has to differ.
    assert main.build_facet_filter(*left, None, None, None, None) != \
        main.build_facet_filter(*right, None, None, None, None)


@pytest.mark.parametrize("left,right", [
    # (author, tag): the same delimiter shift, now across the tag boundary.
    (("b|c", "a"), ("b", "a|c")),
    (("b|c", "a"), ("b|c", "a|b")),
    (("b", "a|c"), ("b", "a|c|")),
])
def test_facet_cache_token_never_collides_across_the_tag_field(left, right):
    assert left != right, "the pair under test must be genuinely different"
    assert main.facet_cache_token(None, None, left[0], None, None, None, left[1]) != \
        main.facet_cache_token(None, None, right[0], None, None, None, right[1])
    # They really do select different filters, so the key has to differ.
    assert main.build_facet_filter(None, None, left[0], None, None, None, left[1]) != \
        main.build_facet_filter(None, None, right[0], None, None, None, right[1])


def test_two_different_tag_filters_get_distinct_cache_entries(monkeypatch):
    """`tag` is part of the cache identity on both legs: reaching only the
    search: key would let the second request be served the first tag's results
    from the retrieve: entry."""
    cache, retrieved, _ = _wire(monkeypatch)

    first = _client.get("/search", params={"q": "test", "tag": "IPO"})
    second = _client.get("/search", params={"q": "test", "tag": "Flipkart"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cached"] is False, "the second tag was served the first one's results"
    assert len(retrieved) == 2, "the second request reused the first one's retrieval"
    search_keys = [k for k in cache.gets if k.startswith("search:")]
    retrieve_keys = [k for k in cache.gets if k.startswith("retrieve:")]
    assert len(set(search_keys)) == 2, "the two tag filters share one search key"
    assert len(set(retrieve_keys)) == 2, "the two tag filters share one retrieve key"
    assert len([k for k in cache.store if k.startswith("search:")]) == 2


def test_build_cache_key_is_injective():
    """A length-prefixed encoding is unambiguous: shifting content between parts
    cannot produce the same key."""
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


# --- no token collision over a corpus of near-miss inputs --------------------

# Injectivity is a property over *many* near-misses at once, not one example, so
# the corpus is driven end to end through the real /search endpoint and the keys
# the real cache received are compared. These are all semantically DIFFERENT
# questions, so every one must keep its own cache entry.
_NEAR_MISS_CORPUS = [
    "fintech funding",
    "Fintech funding",          # case reaches the embedder, so it is not folded
    "fintech funding.",         # trailing punctuation
    "fintech funding?",
    "fintech  funding!",        # collapses to the same text, but differs from all
    "fintech funding 2025",     # an extra token
    "fintech fundin",           # a typo is a different question
    "fintech funcing",
    "fintaech funding",
    "fintech fundingx",
    "fintech funding 2026",     # same length, different year
    "merchant funding",
]


def test_near_miss_corpus_never_collides_on_the_search_cache_key(monkeypatch):
    """Two inputs that hash to one key serve the WRONG cached answer with no
    error anywhere, so every entry of the corpus must get its own key."""
    normalised = {q: normalize_text(q) for q in _NEAR_MISS_CORPUS}
    # Precondition, and the point of the test: normalisation must not merge two
    # genuinely different questions.
    assert len(set(normalised.values())) == len(_NEAR_MISS_CORPUS), (
        "normalisation merged distinct queries: "
        f"{normalised}"
    )

    cache, _retrieved, _recorded = _wire(monkeypatch)
    for q in _NEAR_MISS_CORPUS:
        response = _client.get("/search", params={"q": q})
        assert response.status_code == 200, f"{q!r} did not search: {response.text}"

    keys = [k for k in cache.gets if k.startswith("search:")]
    assert len(keys) == len(_NEAR_MISS_CORPUS), (
        f"expected one search key per request, got {keys}"
    )
    assert len(set(keys)) == len(keys), (
        f"distinct queries shared a cache key: {keys}"
    )
    assert len(cache.sets) == len(_NEAR_MISS_CORPUS)


def test_equivalent_spellings_share_one_search_cache_key(monkeypatch):
    """The other half of the property: if normalisation did nothing every key
    would trivially be distinct, so one question spelled several ways must be
    one entry."""
    spellings = [
        "fintech funding",
        "  fintech   funding  ",
        "ＴＥＳＴ deals",
        "TEST  deals",
        "TEST deals\x00\r\n",
        "ﬁntech funding",
    ]
    distinct_after_normalising = {normalize_text(s) for s in spellings}
    assert distinct_after_normalising == {"fintech funding", "TEST deals"}, (
        "the corpus no longer describes two questions"
    )

    cache, _retrieved, _recorded = _wire(monkeypatch)
    for q in spellings:
        assert _client.get("/search", params={"q": q}).status_code == 200

    keys = [k for k in cache.gets if k.startswith("search:")]
    assert len(set(keys)) == 2, f"expected one entry per question, got {keys}"


def test_a_long_legal_query_is_normalised_and_keyed_within_the_request_clamp(monkeypatch):
    """The seam between the key bound and the request-level clamp: a query long
    enough that it is a digest in the key, short enough that the request is
    legal, must be accepted, canonicalised and keyed once -- with the key
    carrying its digest and none of its text."""
    # Already canonical, so the digest expectation below is about the key rather
    # than about canonicalisation; the variant spelling proves normalisation ran.
    long_q = ("fintech funding roundup " * 12)[:300].strip()
    assert normalize_text(long_q) == long_q
    assert 256 <= len(long_q) <= config.SEARCH_QUERY_MAX_CHARS - 1, len(long_q)
    # The same question in a compatibility spelling: the only way to see that
    # normalisation ran here rather than the key merely being short.
    fullwidth = "".join(chr(ord(c) + 0xFEE0) if "a" <= c <= "z" else c for c in long_q)

    cache, retrieved, _recorded = _wire(monkeypatch)
    ascii_response = _client.get("/search", params={"q": long_q})
    fullwidth_response = _client.get("/search", params={"q": fullwidth})

    assert ascii_response.status_code == 200
    assert fullwidth_response.status_code == 200
    assert fullwidth_response.json()["cached"] is True, "the variant spelling re-ran the pipeline"
    assert len(retrieved) == 1, "the variant spelling re-ran the whole retrieval"

    keys = {k for k in cache.gets if k.startswith("search:")}
    assert len(keys) == 1, f"one question, one entry: {keys}"
    (key,) = keys
    assert len(key) <= MAX_KEY_LEN, f"key grew to {len(key)} chars: {key[:80]!r}"
    assert main._cache_key_component(long_q) in key, "the query must be digested, not spelled out"
    assert long_q[:60] not in key, "the raw query reached the key"

    # The clamp still answers above the window: this bound must not shadow it.
    too_long = _client.get("/search", params={"q": "A" * (config.SEARCH_QUERY_MAX_CHARS + 1)})
    assert too_long.status_code == 422, too_long.status_code


# --- facet bounds -----------------------------------------------------------


def test_facet_value_count_over_the_cap_is_rejected():
    # The input deliberately does NOT scale with MAX_FACET_VALUES: sizing it from
    # the cap means a mutated-up cap makes this allocate a list of that size, so
    # it has to fail fast rather than wedge the suite.
    too_many = ",".join(f"v{i}" for i in range(50))
    with pytest.raises(HTTPException) as excinfo:
        split_facet_values("industry", too_many)
    assert excinfo.value.status_code == 400
    assert "industry" in excinfo.value.detail
    assert str(MAX_FACET_VALUES) in excinfo.value.detail
    # The rejected values are caller input and must not be reflected back.
    assert "v0" not in excinfo.value.detail


def test_facet_value_length_over_the_cap_is_rejected():
    # Fixed length, not MAX_FACET_VALUE_LEN + 1, for the same reason as above.
    with pytest.raises(HTTPException) as excinfo:
        split_facet_values("author", "n" * 400)
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
    assert "industry" in excinfo.value.detail


def test_eleven_values_is_rejected_at_the_shipped_cap():
    """A LITERAL boundary test: cap tests that build their input from the
    constant and assert against the same constant let input and expectation
    scale together, so they pass whatever the constant is."""
    with pytest.raises(HTTPException) as excinfo:
        split_facet_values("industry", ",".join(f"v{i}" for i in range(11)))
    assert excinfo.value.status_code == 400


def test_ten_values_is_accepted_at_the_shipped_cap():
    """The other side: 10 is in bounds, so the cap is a bound and not a blanket
    refusal that would break a real UI selection."""
    assert len(split_facet_values("industry", ",".join(f"v{i}" for i in range(10)))) == 10


def test_a_201_character_value_is_rejected_at_the_shipped_cap():
    """The literal length boundary; scaling from MAX_FACET_VALUE_LEN has the same
    tautology as above."""
    with pytest.raises(HTTPException) as excinfo:
        split_facet_values("author", "n" * 201)
    assert excinfo.value.status_code == 400


def test_a_200_character_value_is_accepted_at_the_shipped_cap():
    assert split_facet_values("author", "n" * 200) == ["n" * 200]


def test_the_longest_tag_in_the_corpus_is_filterable():
    """The cap is set by the corpus, not by a round number: the longest real tag
    is 112 characters, and a filter that cannot name a tag that exists is a dead
    control."""
    longest_tag = "n" * 112
    assert split_facet_values("tag", longest_tag) == [longest_tag]
    filt = main.build_facet_filter(None, None, None, None, None, None, longest_tag)
    assert filt.must[0].key == "tag_names"
    assert filt.must[0].match.any == [longest_tag]


def test_facet_filter_never_builds_an_oversized_match_any():
    """The cap has to hold at the filter itself, not only at the HTTP edge --
    the chat and date-window paths reach build_facet_filter directly."""
    with pytest.raises(HTTPException):
        main.build_facet_filter("y" * 5000, None, None, None, None, None)


def test_facets_at_the_cap_are_still_accepted():
    """The cap must not reject a legitimate in-bounds request. Sized from
    literals, not from MAX_FACET_VALUES / MAX_FACET_VALUE_LEN: this builds the
    product of those two, so a mutated-up cap made it allocate gigabytes."""
    values = ["v" * 200] * 10
    accepted = split_facet_values("industry", ",".join(values))
    assert len(accepted) == 10
    assert all(len(v) == 200 for v in accepted)
    filt = main.build_facet_filter(",".join(values), None, None, None, None, None)
    assert len(filt.must[0].match.any) == 10


def test_oversized_facet_over_http_is_a_400_not_a_200(monkeypatch):
    _wire(monkeypatch)
    bomb = ",".join(f"v{i}" for i in range(5000))
    response = _client.get("/search", params={"q": "test", "industry": bomb})
    assert response.status_code == 400
    assert "industry" in response.json()["detail"]


def test_oversized_facet_never_reaches_a_cache_key(monkeypatch):
    """The /search cache key is built before retrieval runs, so the bound has to
    be applied before the key, not only inside the filter."""
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
    assert value not in excinfo.value.detail
    assert "script" not in excinfo.value.detail
    assert "not-a-date" not in excinfo.value.detail


def test_invalid_date_error_body_is_bounded(monkeypatch):
    """End to end: a long invalid date must produce a small static 400 body.
    Only the retrieval itself is stubbed, so the request really does reach the
    filter builder that validates the date."""
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
    assert cache.sets == []


# --- query normalisation ----------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("ＴＥＳＴ", "TEST"),                      # full-width Latin
    ("ﬁntech", "fintech"),                    # fi ligature
    ("café", "café"),                    # NFD composed by NFKC
    ("test\x00injected", "testinjected"),      # NUL removed
    ("test\r\nINJECTED", "test INJECTED"),    # CRLF -> space, never fused
    ("test\x01INJECTED", "testINJECTED"),     # other C0 removed
    ("Ola\tIPO", "Ola IPO"),                  # a tab separates words
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
    cache key, the analytics record or the echoed response."""
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
    """The /search key is a second, independent key a query reaches, so it has to
    be bounded on its own terms: covering only the retrieve key would leave a
    long query to build an arbitrarily long /search key.

    Two bounds compose here -- the query segment is digested per
    CACHE_KEY_QUERY_MAX_CHARS, the whole key once it passes MAX_KEY_LEN -- and a
    query long enough to trip the second but short enough to be digested by the
    first must never spell itself out either way."""
    over = MAX_KEY_LEN
    under_clamp = config.SEARCH_QUERY_MAX_CHARS - 1
    assert over < under_clamp, (
        "this test needs a query long enough to trip the key bound but short "
        "enough to pass the request-level clamp; raise MAX_KEY_LEN or the clamp"
    )
    cache, _, _ = _wire(monkeypatch)
    response = _client.get("/search", params={"q": "q" * over})
    assert response.status_code == 200
    assert cache.gets, "expected the request to build a cache key"
    digest = main._cache_key_component("q" * over)
    assert digest.startswith("h:"), "the query is long enough to be digested"
    # The same request also builds the retrieve: prefetch key, so the search key
    # is selected by namespace rather than assumed to be the only one read.
    search_keys = [k for k in cache.gets if k.startswith("search:")]
    assert search_keys, "expected the request to build a search cache key"
    for key in search_keys:
        assert len(key) <= MAX_KEY_LEN, f"key grew to {len(key)} chars: {key[:80]!r}"
        assert digest in key or key.startswith("search:sha256:")
        assert "q" * 100 not in key, "the raw query reached the key"


def test_search_cache_key_stays_bounded_with_a_max_sized_facet(monkeypatch):
    """A long-but-in-bounds facet set must not push the key past the bound
    either. This one stays under the limit and must remain readable rather than
    being digested for nothing."""
    cache, _, _ = _wire(monkeypatch)
    # Literals, not the cap constants: this multiplies the two together.
    values = ",".join("v" * 200 for _ in range(10))
    response = _client.get("/search", params={"q": "test", "industry": values})
    assert response.status_code == 200
    # Both the search: and retrieve: keys must stay bounded.
    search_keys = [k for k in cache.gets if k.startswith("search:")]
    assert search_keys, "expected the request to build a search cache key"
    for key in cache.gets:
        assert len(key) <= MAX_KEY_LEN, f"key grew to {len(key)} chars: {key[:80]!r}"
    # A max-sized facet is ~1 KB, so the facet component is the part that has to
    # be digested rather than echoed as caller text.
    for key in search_keys:
        assert ":sha256:" in key, "a max-sized facet must be digested, not spelled out"
        assert "v" * 200 not in key, "the raw facet reached the key"


def test_retrieve_cache_key_is_bounded_and_control_free(monkeypatch):
    """The retrieve-level key is the other key a query reaches (chat shares this
    pipeline), so it is bounded and cleaned on the same terms."""
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


def test_click_analytics_key_carries_no_control_characters():
    """The click beacon is unauthenticated, so its query is the least trusted
    string in the app: it reaches Redis as a keyed digest taken over the
    normalised form, so a NUL/CRLF can neither appear in the key nor split one
    query into two keys."""
    key = analytics._click_query_key("test\x00\r\nINJECTED: admin", "k")
    assert "\x00" not in key
    assert "\r" not in key and "\n" not in key
    assert "INJECTED" not in key


def test_click_analytics_key_aggregates_equivalent_spellings():
    assert analytics._click_query_key("ＴＥＳＴ deals", "k") == \
        analytics._click_query_key("TEST deals", "k")


def test_click_analytics_key_stays_length_bounded():
    """Canonicalising must not have displaced the bound on the unauthenticated
    beacon."""
    key = analytics._click_query_key("q" * 100_000, "k")
    assert len(key) < 1000


def test_click_analytics_key_is_scoped_by_the_digest_key():
    """Two deployments must not read each other's click signal, so the secret is
    mixed in rather than the key being a bare hash of the query."""
    assert analytics._click_query_key("TEST deals", "k1") != \
        analytics._click_query_key("TEST deals", "k2")


class _MemberRedis:
    """Records the ZSET members analytics actually writes."""

    def __init__(self):
        self.members: list[tuple[str, str]] = []

    def pipeline(self):
        return self

    def zincrby(self, key, amount, member):
        self.members.append((key, member))
        return self

    def incr(self, *a, **k):
        return self

    def expire(self, *a, **k):
        return self

    async def execute(self):
        return []


def _use_fixed_digest_key(monkeypatch, fake):
    """Pin the analytics digest secret so the recorders take the keyed path
    without needing a real Redis to persist a generated key into."""
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics.config, "ANALYTICS_QUERY_KEY", "test-digest-key")
    monkeypatch.setattr(analytics, "_QUERY_DIGEST_KEY", None)


@pytest.mark.parametrize("call", ["search", "click"])
def test_analytics_sorted_set_members_carry_no_user_text(monkeypatch, call):
    """The top-query ZSETs store a keyed digest, never the query. The real
    recorders are driven here because the property is about the bytes on the
    wire, not about a helper's return value."""
    fake = _MemberRedis()
    _use_fixed_digest_key(monkeypatch, fake)
    raw = "ＴＥＳＴ deals\x00\r\n"
    if call == "search":
        asyncio.run(analytics.record_search(raw, 1, False, False, 10.0, False))
        target = "analytics:top_queries"
    else:
        asyncio.run(analytics.record_click(raw, 0, 7))
        # record_click writes two members: the query digest in click_top_queries,
        # and the article id in the per-query set.
        target = "analytics:click_top_queries"

    written = [m for key, m in fake.members if key == target]
    assert written, f"expected a member under {target}"
    for member in written:
        assert "\x00" not in member and "\r" not in member and "\n" not in member
        # Not merely scrubbed: the query is not recoverable from the member.
        assert "deals" not in member.lower()
        assert member == analytics.query_digest(raw, analytics._QUERY_DIGEST_KEY)


def test_equivalent_spellings_aggregate_into_one_top_query_row(monkeypatch):
    """The point of canonicalising the member, not merely hiding it: two
    spellings of one query are one row, so the list is not split by
    presentation."""
    fake = _MemberRedis()
    _use_fixed_digest_key(monkeypatch, fake)
    spellings = ("ＴＥＳＴ deals", "TEST  deals", "TEST deals\x00")
    for spelling in spellings:
        asyncio.run(analytics.record_search(spelling, 1, False, False, 10.0, False))
    members = [m for key, m in fake.members if key == "analytics:top_queries"]
    assert len(set(members)) == 1, f"split across rows: {members}"
    assert members[0] == analytics.query_digest("TEST deals", analytics._QUERY_DIGEST_KEY)

