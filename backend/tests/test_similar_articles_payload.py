"""The recommendation feeds must not ship article bodies: neither the wire nor the cache may carry one."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import main, recommender
from app.config import config

# Every field the UI reads; mirrors frontend/app/components/SimilarArticles.tsx and frontend/app/for-you/page.tsx.
UI_FIELDS = {
    "id",
    "title",
    "url",
    "published_date",
    "category",
    "summary",
    "industry_names",
}

WHOLE_PAYLOAD = True

# The queried article: ``_format_articles`` excludes it, so a fixture point must not use this id.
SOURCE_ID = 7
OTHER_ID = 8


def _run(coro):
    return asyncio.run(coro)


def _body():
    """A body at the configured ceiling -- the worst case that would ship."""
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
    """An entry in the old shape: still carrying the body."""
    return [{"id": SOURCE_ID, "title": "stale", "url": "https://x/y", "body": _body()}]


def _user_request(user_id="user-1"):
    return SimpleNamespace(state=SimpleNamespace(user_id=user_id))


def _stale_personalization(monkeypatch):
    """main and recommender import get_user_interactions separately; patch both or one reaches live Redis."""

    async def fake_interactions(user_id):
        return [(SOURCE_ID, "click")]

    async def fake_categories(user_id):
        return [("technology", 1.0)]

    monkeypatch.setattr(main, "get_user_interactions", fake_interactions)
    monkeypatch.setattr(recommender, "get_user_interactions", fake_interactions)
    monkeypatch.setattr(recommender, "get_user_profile_categories", fake_categories)


@pytest.fixture
def wired(monkeypatch):
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
        _cache, client = wired
        stored = client.query_points.return_value.points
        assert all(len(p.payload["body"]) == config.BODY_CHAR_LIMIT for p in stored)

        response = _call_similar()

        assert response.similar_articles, "expected similar articles to be returned"
        for article in response.similar_articles:
            assert "body" not in article, f"body leaked into response: {sorted(article)}"

    def test_response_still_carries_every_field_the_ui_renders(self, wired):
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
        first = _call_similar()
        second = _call_similar()

        assert second.cached is True, "expected the second call to be served from cache"
        assert first.similar_articles == second.similar_articles
        assert second.similar_articles
        for article in second.similar_articles:
            assert "body" not in article

    def test_body_free_when_the_point_carries_only_a_title_and_url(self, wired):
        _cache, client = wired
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
        """The endpoint returns its cached value verbatim, so a body-bearing cache key must never be read."""
        cache, _client = wired
        cache.store[f"recommend:similar:{SOURCE_ID}:3:False"] = _legacy_entry()

        response = _call_similar()

        assert response.cached is False, "served a similar entry written by the previous shape"
        assert response.similar_articles, "expected a fresh fetch instead"
        for article in response.similar_articles:
            assert "body" not in article

    def test_legacy_for_you_entry_is_not_served(self, wired, monkeypatch):
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
        _cache, client = wired
        _call_similar()

        selector = client.query_points.call_args.kwargs["with_payload"]
        assert selector is not WHOLE_PAYLOAD
        assert isinstance(selector, list), f"expected a field list, got {type(selector).__name__}"
        assert "body" not in selector

    def test_every_field_the_ui_renders_is_requested(self, wired):
        _cache, client = wired
        _call_similar()
        requested = set(client.query_points.call_args.kwargs["with_payload"])

        for field in ("title", "url", "published_date", "category", "summary", "industry_names"):
            assert field in requested, f"{field} is rendered by the UI but not requested"

    def test_all_recommender_payload_requests_stop_asking_for_bodies(self, wired, monkeypatch):
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


# The search page renders one <SimilarArticles> per result, so top_k=8 fires eight requests.
RESULTS_PER_SEARCH = 8
SIMILAR_LIMIT = 3
# Qdrant is asked for limit*3 with the source in must_not, and nothing truncates afterwards.
ARTICLES_PER_REQUEST = SIMILAR_LIMIT * 3


class TestMeasuredPageViewPayload:
    """The total is measured through the real handler, so a returning ``body`` shows up as more bytes."""

    def _page_view_bytes(self, monkeypatch):
        cache = _RecordingCache()
        monkeypatch.setattr(main, "cache", cache)

        # Enough distinct points that all eight queries skip must_not and still fill.
        points = [
            _point(SOURCE_ID + i, body=_body())
            for i in range(RESULTS_PER_SEARCH + ARTICLES_PER_REQUEST)
        ]
        # The fake applies `limit` and `must_not` server-side, as the real Qdrant does.
        client = _qdrant(points)
        real_query_points = client.query_points

        async def query_points(**kwargs):
            excluded = {
                cond.match.value
                for cond in (kwargs.get("query_filter").must_not if kwargs.get("query_filter") else [])
            }
            allowed = [p for p in points if p.id not in excluded]
            response = await real_query_points(**kwargs)
            response.points = allowed[: kwargs["limit"]]
            return response

        client.query_points = AsyncMock(side_effect=query_points)
        monkeypatch.setattr(recommender, "state", {"qdrant": client})

        total = 0
        articles = 0
        for i in range(RESULTS_PER_SEARCH):
            # A distinct source id per result means a distinct cache key, so no request is a cache hit.
            response = _run(
                main.get_similar(
                    article_id=SOURCE_ID + i,
                    limit=SIMILAR_LIMIT,
                    same_category=False,
                )
            )
            total += len(response.model_dump_json().encode())
            articles += len(response.similar_articles)

        return total, articles

    def test_a_top_k_8_page_view_serializes_far_below_the_bodies_it_stores(self, wired, monkeypatch):
        measured, articles = self._page_view_bytes(monkeypatch)

        assert articles == RESULTS_PER_SEARCH * ARTICLES_PER_REQUEST, (
            f"expected {RESULTS_PER_SEARCH * ARTICLES_PER_REQUEST} articles per page view, "
            f"got {articles}; the measurement no longer models the endpoint"
        )

        stored_body_bytes = articles * config.BODY_CHAR_LIMIT

        # A body-bearing build puts 72 x 50k characters here, so 100 kB separates the two.
        assert measured < 100_000, f"a top_k=8 page view serialized to {measured} B"

        assert measured < stored_body_bytes / 10, (
            f"page view is {measured} B against {stored_body_bytes} B of stored bodies"
        )
