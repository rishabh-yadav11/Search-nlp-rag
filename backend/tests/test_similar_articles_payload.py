"""The recommendation feeds must not ship article bodies (#257).

``/recommend/similar/{id}`` used to ask Qdrant for whole points
(``with_payload=True``) and copy ``body`` -- up to ``BODY_CHAR_LIMIT`` (50k)
chars per point -- into every returned dict. The response model is untyped, so
nothing stripped it: the bodies went over the wire, into the Redis cache for
an hour, and back out on every subsequent read. The search page renders one
``SimilarArticles`` per result, so a single top_k=8 search dragged >3MB per
page view to display a title and a category.

Measured with bodies at ``BODY_CHAR_LIMIT`` and the real ``_format_articles``
-- 9 points fetched per request, 8 returned once the source article is
filtered out, 8 requests for a top_k=8 search -- one page view went from
3,225,178 B to 24,410 B. The before-figure is robust (it is dominated by
eight 50k bodies); the after-figure moves with the synthetic title/summary
text of the measuring fixture, so independent recomputes land between
~21 kB and ~24 kB. Either way >3MB per view is gone.

Nothing renders the body. ``SimilarArticles.tsx`` reads id/title/url/category,
summary and published_date; the for-you card reads the same plus
industry_names. These tests pin that contract from both ends: the wire and the
cache stay body-free, the display fields survive, the Qdrant requests
themselves are narrowed so the bodies are never even transferred, and a
cache entry written by the pre-fix shape is never served.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import main, recommender
from app.config import config

# Every field the UI reads off a recommendation object. Mirrors
# frontend/app/components/SimilarArticles.tsx (its Article interface plus the
# compact and full render) and frontend/app/for-you/page.tsx.
UI_FIELDS = {
    "id",
    "title",
    "url",
    "published_date",
    "category",
    "summary",
    "industry_names",
}

# The request shape that used to drag the bodies over the wire.
WHOLE_PAYLOAD = True

# The article the fixtures query "about". _format_articles excludes it, so a
# fixture point must NOT use this id if the test needs a returned article.
SOURCE_ID = 7
# A second id that survives exclusion.
OTHER_ID = 8


def _run(coro):
    return asyncio.run(coro)


def _body():
    """A body at the configured ceiling -- the worst case that used to ship."""
    return ("VCCircle deal coverage. " * 4000)[: config.BODY_CHAR_LIMIT]


def _point(pid, body=""):
    point = MagicMock()
    point.id = pid
    point.score = 0.9
    point.payload = {
        "title": f"Acme Corp raises a round ({pid})",
        "url": f"https://www.vccircle.com/deal/{pid}",
        "published_date": "2026-09-01T10:00:00+00:00",
        "category": "Private Equity",
        "summary": "Acme Corp raised a round led by Big Fund.",
        "author_names": ["Jane Reporter"],
        "industry_names": ["Technology"],
        "dealtype_names": ["Fund Raising"],
        "content_type": "deals",
        "body": body,
    }
    return point


def _qdrant(points):
    client = MagicMock()
    response = MagicMock()
    response.points = points
    client.query_points = AsyncMock(return_value=response)
    client.retrieve = AsyncMock(return_value=list(points))
    client.scroll = AsyncMock(return_value=(list(points), None))
    return client


class _RecordingCache:
    """In-memory stand-in for the HybridCache that keeps what was written."""

    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)

    async def delete_prefix(self, prefix):
        for key in [k for k in self.store if k.startswith(prefix)]:
            del self.store[key]


def _legacy_entry():
    """An entry as the pre-fix shape wrote it: still carrying the body."""
    return [{"id": SOURCE_ID, "title": "stale", "url": "https://x/y", "body": _body()}]


def _user_request(user_id="user-1"):
    return SimpleNamespace(state=SimpleNamespace(user_id=user_id))


def _stale_personalization(monkeypatch):
    """Give a user one interaction, hermetically.

    ``main`` and ``recommender`` each import ``get_user_interactions``
    separately, so both namespaces are patched; otherwise the recommender copy
    reaches a real Redis and the test's outcome depends on whether one happens
    to be running.
    """

    async def fake_interactions(user_id):
        return [(SOURCE_ID, "click")]

    async def fake_categories(user_id):
        return [("technology", 1.0)]

    monkeypatch.setattr(main, "get_user_interactions", fake_interactions)
    monkeypatch.setattr(recommender, "get_user_interactions", fake_interactions)
    monkeypatch.setattr(recommender, "get_user_profile_categories", fake_categories)


@pytest.fixture
def wired(monkeypatch):
    """Wire a fake Qdrant + cache into the real recommender and endpoints."""
    cache = _RecordingCache()
    points = [_point(SOURCE_ID, body=_body()), _point(OTHER_ID, body=_body())]
    client = _qdrant(points)
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(recommender, "state", {"qdrant": client})

    async def fake_trending(limit, *args, **kwargs):
        return [{"article_id": SOURCE_ID, "score": 3.0}]

    monkeypatch.setattr(recommender, "get_trending_articles", fake_trending)
    return cache, client


def _call_similar(article_id=SOURCE_ID, limit=3, same_category=False):
    return _run(main.get_similar(article_id=article_id, limit=limit, same_category=same_category))


def _all_payload_selectors(client):
    selectors = [c.kwargs["with_payload"] for c in client.query_points.call_args_list]
    selectors += [c.kwargs["with_payload"] for c in client.retrieve.call_args_list]
    selectors += [c.kwargs["with_payload"] for c in client.scroll.call_args_list]
    return selectors


class TestSimilarResponseExcludesBody:
    def test_response_has_no_body_even_though_stored_points_carry_one(self, wired):
        """The stored points still hold full 50k bodies; none may reach the wire."""
        _cache, client = wired
        stored = client.query_points.return_value.points
        assert all(len(p.payload["body"]) == config.BODY_CHAR_LIMIT for p in stored)

        response = _call_similar()

        assert response.similar_articles, "expected similar articles to be returned"
        for article in response.similar_articles:
            assert "body" not in article, f"body leaked into response: {sorted(article)}"

    def test_response_still_carries_every_field_the_ui_renders(self, wired):
        """Dropping the body must not take any rendered field with it."""
        response = _call_similar()

        assert response.similar_articles
        for article in response.similar_articles:
            assert UI_FIELDS <= set(article), f"UI fields missing: {UI_FIELDS - set(article)}"
            assert article["title"].startswith("Acme Corp")
            assert article["url"] == f"https://www.vccircle.com/deal/{article['id']}"
            assert article["category"] == "Private Equity"
            assert article["industry_names"] == ["Technology"]
            assert article["published_date"] == "2026-09-01T10:00:00+00:00"

    def test_cached_response_is_equally_body_free(self, wired):
        """The second read comes from Redis and must not reintroduce the body."""
        first = _call_similar()
        second = _call_similar()

        assert second.cached is True, "expected the second call to be served from cache"
        assert first.similar_articles == second.similar_articles
        assert second.similar_articles
        for article in second.similar_articles:
            assert "body" not in article

    def test_body_free_when_the_point_carries_only_a_title_and_url(self, wired):
        """A point with no optional fields must not smuggle a body back in."""
        _cache, client = wired
        # OTHER_ID, not SOURCE_ID: the source is filtered out, which would
        # leave nothing to assert on.
        client.query_points.return_value.points = [_point(OTHER_ID, body=_body())]
        client.query_points.return_value.points[0].payload = {
            "title": "Bare point",
            "url": "https://example.com/x",
            "body": _body(),
        }

        response = _call_similar()

        assert len(response.similar_articles) == 1, "the bare point should still be returned"
        assert "body" not in response.similar_articles[0]


class TestCacheEntriesExcludeBody:
    def test_redis_entry_holds_no_body(self, wired):
        cache, _client = wired
        _call_similar()

        assert cache.store, "expected the endpoint to cache its result"
        ((key, value),) = cache.store.items()
        assert key.startswith("recommend:similar:")
        assert value, "cached value should not be empty"
        for article in value:
            assert "body" not in article, f"body persisted to Redis under {key}"

    def test_cached_entry_keeps_the_rendered_fields(self, wired):
        cache, _client = wired
        _call_similar()

        ((_key, value),) = cache.store.items()
        # The source article is filtered out, so the two stored points yield one.
        assert len(value) == 1
        for article in value:
            assert UI_FIELDS <= set(article)

    def test_for_you_cache_entry_holds_no_body(self, wired, monkeypatch):
        """for-you shares the formatter and has its own 30-minute cache."""
        cache, _client = wired
        _stale_personalization(monkeypatch)

        response = _run(main.get_for_you(limit=3, _auth=None, request=_user_request()))

        assert response.recommendations, "expected for-you to return recommendations"
        assert cache.store, "expected for-you to cache its result"
        for value in cache.store.values():
            assert value
            for article in value:
                assert "body" not in article

    def test_legacy_similar_entry_is_not_served(self, wired):
        """A pre-deploy entry still holds the bodies; the endpoint returns cache
        verbatim, so it must not read a key written by the old shape."""
        cache, _client = wired
        cache.store[f"recommend:similar:{SOURCE_ID}:3:False"] = _legacy_entry()

        response = _call_similar()

        assert response.cached is False, "served a similar entry written by the previous shape"
        assert response.similar_articles, "expected a fresh fetch instead"
        for article in response.similar_articles:
            assert "body" not in article

    def test_legacy_for_you_entry_is_not_served(self, wired, monkeypatch):
        """for-you returns its cached value verbatim too, so its key is versioned."""
        cache, _client = wired
        _stale_personalization(monkeypatch)
        cache.store["recommend:for-you:user-1:3"] = _legacy_entry()

        response = _run(main.get_for_you(limit=3, _auth=None, request=_user_request()))

        assert response.cached is False, "served a for-you entry written by the previous shape"
        assert response.recommendations
        for article in response.recommendations:
            assert "body" not in article
            assert article.get("title") != "stale"

    def test_legacy_trending_entry_is_not_served(self, wired):
        """trending shares the formatter and returns its cached value verbatim."""
        cache, _client = wired
        cache.store["recommend:trending:3"] = _legacy_entry()

        response = _run(main.get_trending(limit=3, _auth=None))

        # TrendingResponse carries no `cached` flag, so prove it by what came back.
        assert response.articles, "expected a fresh trending fetch"
        for article in response.articles:
            assert "body" not in article
            assert article.get("title") != "stale", "served a trending entry written by the previous shape"


class TestQdrantRequestIsNarrowed:
    def test_similar_query_requests_a_field_list_not_the_whole_payload(self, wired):
        """with_payload=True is the root cause: it transfers 50k chars per point."""
        _cache, client = wired
        _call_similar()

        selector = client.query_points.call_args.kwargs["with_payload"]
        assert selector is not WHOLE_PAYLOAD
        assert isinstance(selector, list), f"expected a field list, got {type(selector).__name__}"
        assert "body" not in selector

    def test_every_field_the_ui_renders_is_requested(self, wired):
        """A field the UI renders but the query never requests renders blank."""
        _cache, client = wired
        _call_similar()
        requested = set(client.query_points.call_args.kwargs["with_payload"])

        for field in ("title", "url", "published_date", "category", "summary", "industry_names"):
            assert field in requested, f"{field} is rendered by the UI but not requested"

    def test_all_recommender_payload_requests_stop_asking_for_bodies(self, wired, monkeypatch):
        """for-you / trending / latest share the formatter; none renders a body."""
        _cache, client = wired
        _stale_personalization(monkeypatch)
        _call_similar()
        _run(main.get_for_you(limit=3, _auth=None, request=_user_request()))
        _run(main.get_trending(limit=3, _auth=None))

        selectors = _all_payload_selectors(client)
        assert selectors, "expected at least one Qdrant payload request"
        for selector in selectors:
            assert selector is not WHOLE_PAYLOAD, "with_payload=True re-introduces the body transfer"
            assert "body" not in selector, f"body re-requested via {selector}"
