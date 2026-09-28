"""Tests for the batched similar-articles route (#353).

A search view renders one ``<SimilarArticles>`` per result, so before this
route existed that view cost one HTTP request, one cache read and one Qdrant
point-id query per result -- eight of each for a ``top_k=8`` page. These
tests pin the properties that make the batched form cheaper rather than just
differently shaped:

* a whole view is read with ONE cache command, not one per article;
* the Qdrant work is unchanged (one query per uncached id) and a warm view
  spends none, so the saving is round trips, not a different algorithm;
* the batched and per-article routes share one cache key, so a view warmed
  through either is a hit for the other.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app import main
from app.redis_cache import HybridCache

USER = "user-similar-batch"


class _StubRedis:
    """Async Redis that records every command the cache issues."""

    def __init__(self):
        self.store = {}
        self.gets = []
        self.mgets = []
        self.sets = []

    async def get(self, key):
        self.gets.append(key)
        return self.store.get(key)

    async def mget(self, keys, *_rest):

        # Production calls this BOTH ways: redis_cache.py:148 `mget(keys)` and

        # :201 `mget(*keys)`. Accept either shape rather than pinning one.

        if isinstance(keys, str):

            keys = [keys, *_rest]

        else:

            keys = [*keys, *_rest]
        self.mgets.append(list(keys))
        return [self.store.get(key) for key in keys]

    async def set(self, key, value, ex=None):
        self.sets.append((key, ex))
        self.store[key] = value
        return True

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)

    async def aclose(self):
        return None

    @property
    def commands(self):
        """Total commands issued -- the quantity the batch route exists to cut."""
        return len(self.gets) + len(self.mgets) + len(self.sets)


def _near(article_id):
    return {
        "id": article_id * 100,
        "title": f"near {article_id}",
        "url": f"https://vccircle.com/near/{article_id}",
    }


def _key(article_id, limit=3, same_category=False):
    """The production cache key, spelled out here.

    The version is read from the module rather than pasted, so bumping it does
    not break this file; the rest is written out, so a change to the rest of
    the key is visible here instead of being quietly absorbed.
    """
    version = main.RECOMMEND_CACHE_VERSION
    return f"recommend:similar:{version}:{article_id}:{limit}:{same_category}"


def _seed(store, article_id, limit=3, same_category=False):
    """Warm one entry the way a real Redis holds it: as a JSON string."""
    store[_key(article_id, limit, same_category)] = json.dumps([_near(article_id)])


@pytest.fixture
def env(monkeypatch):
    """A real HybridCache over a recording Redis, and a recording recommender.

    The vector search itself is stubbed: what these tests are about is how many
    times the route reaches for it, not what a nearest-neighbour search
    returns. What they assert is the routing -- which ids were computed, which
    came from cache, and how many commands it took to find out.
    """
    redis_stub = _StubRedis()
    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=1000, max_bytes=1 << 24)
    cache._redis = redis_stub
    monkeypatch.setattr(main, "cache", cache)

    computed = []

    async def _similar_articles(article_id, limit=3, same_category=False, **kwargs):
        computed.append(article_id)
        return [_near(article_id)] * limit

    monkeypatch.setattr(main, "get_similar_articles", _similar_articles)
    monkeypatch.setattr(main.config, "ENABLE_RECOMMENDATIONS", True)

    return SimpleNamespace(redis=redis_stub, computed=computed)


@pytest.fixture
def client(monkeypatch):
    """TestClient with the auth gate bypassed, so the handler is reachable."""

    async def _authed(request: Request) -> None:
        request.state.user_id = USER

    monkeypatch.setitem(main.app.dependency_overrides, main.require_auth, _authed)
    return TestClient(main.app)


def _groups(response):
    body = response.json()
    return {group["article_id"]: group for group in body["results"]}


def test_a_whole_view_is_read_from_cache_in_one_command(client, env):
    """Eight articles, one cache command -- not eight.

    The single command is the whole point of the batch: reading the view's
    cached lists one key at a time is what the N+1 looked like from inside
    the server, and no amount of client-side batching changes that.
    """
    ids = list(range(1, 9))
    response = client.post("/recommend/similar/batch", json={"article_ids": ids, "limit": 3})

    assert response.status_code == 200, response.text
    assert env.redis.mgets == [[_key(i) for i in ids]]
    assert env.redis.gets == [], "a batched read must not fall back to per-key GETs"

    groups = _groups(response)
    assert [group["article_id"] for group in response.json()["results"]] == ids, (
        "results must come back in the order they were asked for"
    )
    assert all(len(group["similar_articles"]) == 3 for group in groups.values())


def test_a_batch_costs_fewer_cache_commands_than_the_per_article_route(client, env):
    """The comparison the issue is actually about, measured on both routes."""
    ids = list(range(1, 9))

    for article_id in ids:
        assert client.get(f"/recommend/similar/{article_id}", params={"limit": 3}).status_code == 200
    per_article_commands = env.redis.commands

    env.redis.gets.clear()
    env.redis.mgets.clear()
    env.redis.sets.clear()
    env.redis.store.clear()
    env.computed.clear()

    assert client.post(
        "/recommend/similar/batch", json={"article_ids": ids, "limit": 3}
    ).status_code == 200
    batched_commands = env.redis.commands

    assert batched_commands < per_article_commands, (
        f"batching must spend fewer cache commands than the per-article route "
        f"({batched_commands} vs {per_article_commands})"
    )
    # The vector work is deliberately NOT reduced: batching is a request-shape
    # fix, and a test that let the Qdrant call count silently drop with it
    # would be asserting a cheaper algorithm that does not exist.
    assert env.computed == ids, "each uncached id still costs exactly one query"


def test_a_warm_view_answers_without_any_qdrant_query(client, env):
    """A view whose lists are all cached must spend nothing on Qdrant."""
    ids = [1, 2, 3]
    for article_id in ids:
        _seed(env.redis.store, article_id)

    response = client.post("/recommend/similar/batch", json={"article_ids": ids, "limit": 3})

    assert response.status_code == 200, response.text
    assert env.computed == [], "a warm view must not re-run the vector query"
    groups = _groups(response)
    assert set(groups) == set(ids)
    assert all(group["cached"] is True for group in groups.values())


def test_only_the_uncached_ids_are_queried(client, env):
    """A partly warm view only pays for what is actually missing."""
    _seed(env.redis.store, 1)

    response = client.post(
        "/recommend/similar/batch", json={"article_ids": [1, 2, 3], "limit": 3}
    )

    assert response.status_code == 200, response.text
    assert env.computed == [2, 3], "only the missing ids may reach the vector store"
    groups = _groups(response)
    assert groups[1]["cached"] is True
    assert groups[2]["cached"] is False
    assert [group["article_id"] for group in response.json()["results"]] == [1, 2, 3], (
        "a mixed warm/cold view must still answer in request order"
    )


def test_a_repeated_id_is_queried_once(client, env):
    """The same article listed twice costs one query and one group."""
    response = client.post(
        "/recommend/similar/batch", json={"article_ids": [7, 7, 8, 7], "limit": 3}
    )

    assert response.status_code == 200, response.text
    assert env.computed == [7, 8]
    assert [group["article_id"] for group in response.json()["results"]] == [7, 8]


def test_the_batched_and_per_article_routes_share_one_cache_key(client, env):
    """A view warmed through either route must be a hit for the other.

    Two spellings of the same key would each miss the other's entries, so every
    navigation back to a page would re-run the Qdrant query the cache exists
    to avoid -- the exact cost this issue exists to remove, reintroduced
    through a one-character difference in a format string.
    """
    assert client.get("/recommend/similar/4", params={"limit": 3}).status_code == 200
    env.computed.clear()

    response = client.post("/recommend/similar/batch", json={"article_ids": [4], "limit": 3})

    assert response.status_code == 200, response.text
    assert env.computed == [], "the per-article route's entry must satisfy the batch"
    assert _groups(response)[4]["cached"] is True


def test_an_empty_result_is_not_cached(client, env, monkeypatch):
    """Parity with the per-article route: no rows means no hour-long entry.

    An article with nothing similar is usually a transient Qdrant miss, and
    caching that emptiness would make the view render blank for the whole TTL.
    """
    async def _no_rows(article_id, limit=3, same_category=False, **kwargs):
        return []

    monkeypatch.setattr(main, "get_similar_articles", _no_rows)

    first = client.post("/recommend/similar/batch", json={"article_ids": [5], "limit": 3})
    second = client.post("/recommend/similar/batch", json={"article_ids": [5], "limit": 3})

    assert first.status_code == 200 and second.status_code == 200
    assert _groups(first)[5]["similar_articles"] == []
    assert env.redis.sets == [], "an empty result must not be written to the cache"


def test_a_legacy_unversioned_entry_is_not_served(client, env):
    """The batch route must honour the payload version too, not just the single one.

    ``test_similar_articles_payload`` pins this for ``/recommend/similar/{id}``
    (#257): entries written before the payload narrowed still carry the full
    article body, and the handler returns a cached value verbatim. The batched
    route reads a cache too, so a key of its own that omitted the version
    would serve those bodies for the rest of their TTL -- a regression that
    only the batched surface would have had, and that no other test here
    would notice.
    """
    env.redis.store["recommend:similar:3:3:False"] = json.dumps(
        [{"id": 3, "title": "stale", "url": "https://x/y", "body": "x" * 4000}]
    )

    response = client.post("/recommend/similar/batch", json={"article_ids": [3], "limit": 3})

    assert response.status_code == 200, response.text
    group = _groups(response)[3]
    assert group["cached"] is False, "served a similar entry written by the previous shape"
    assert group["similar_articles"], "expected a fresh fetch instead"
    for article in group["similar_articles"]:
        assert "body" not in article


def test_a_different_limit_is_a_different_cache_entry(client, env):
    """The limit is part of the key, exactly as it is for the per-article route."""
    assert client.get("/recommend/similar/6", params={"limit": 3}).status_code == 200
    env.computed.clear()

    response = client.post("/recommend/similar/batch", json={"article_ids": [6], "limit": 5})

    assert response.status_code == 200, response.text
    assert env.computed == [6], "a different limit is a different question"


def test_same_category_is_part_of_the_cache_key(client, env):
    assert client.get("/recommend/similar/9", params={"limit": 3}).status_code == 200
    env.computed.clear()

    response = client.post(
        "/recommend/similar/batch",
        json={"article_ids": [9], "limit": 3, "same_category": True},
    )

    assert response.status_code == 200, response.text
    assert env.computed == [9], "a same-category batch must not answer from a mixed cache entry"


def test_an_empty_batch_is_rejected(client, env):
    response = client.post("/recommend/similar/batch", json={"article_ids": []})

    assert response.status_code == 422, response.text
    assert env.computed == []


def test_a_batch_over_the_id_cap_is_rejected(client, env):
    """The cap is what stops one request from asking for a whole page, repeatedly."""
    ids = list(range(1, main.SIMILAR_BATCH_MAX_IDS + 2))

    response = client.post("/recommend/similar/batch", json={"article_ids": ids})

    assert response.status_code == 422, response.text
    assert env.computed == [], "an over-cap batch must be refused before it reaches Qdrant"


def test_the_batch_route_requires_authentication(env):
    """Same gate as the per-article route: these are the user's own articles."""
    response = TestClient(main.app).post(
        "/recommend/similar/batch", json={"article_ids": [1, 2]}
    )

    assert response.status_code == 401, response.text
    assert env.computed == []
