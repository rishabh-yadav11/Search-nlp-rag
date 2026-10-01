"""Tests for the /recommend/for-you cold-start contract.

``/recommend/for-you`` is auth-gated: ``require_auth`` either raises 401 or
installs a real user id or ``SERVICE_USER_ID``, so there is no anonymous
response to return. The contract these tests pin:

* an authenticated user with no interactions still gets a populated
  ``cold_start=True`` response, so removing the recommender's cold-start
  fallback goes red;
* a request with no credentials is rejected at the auth gate, before any
  handler logic;
* for every id the auth layer can install, the response is attributed to that
  id and never to ``"anonymous"``.
"""

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app import auth, main, recommender
from app.redis_cache import HybridCache

USER = "user-cold-start"
TOP_STORIES = [{"id": 11, "title": "latest one"}, {"id": 12, "title": "latest two"}]


class _StubRedis:
    """Minimal async Redis. The cold-start path only needs get/set/delete."""

    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)

    def scan_iter(self, match=None, count=None):
        async def _walk():
            for key in list(self.store):
                yield key

        return _walk()

    async def aclose(self):
        return None


@pytest.fixture
def env(monkeypatch):
    """Route the real recommender at a controlled Redis/Qdrant boundary.

    ``get_personalized_recommendations`` is deliberately NOT stubbed: it owns
    cold-start, so stubbing it would make these tests assert their own fixture.
    Only the I/O it reaches through is replaced.
    """
    redis_stub = _StubRedis()
    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=1000, max_bytes=1 << 24)
    cache._redis = redis_stub
    monkeypatch.setattr(main, "cache", cache)

    async def _no_interactions(user_id):
        return []

    async def _no_categories(user_id):
        return []

    async def _latest_top_stories(limit, exclude_ids=None):
        return list(TOP_STORIES)[:limit]

    monkeypatch.setattr(main, "get_user_interactions", _no_interactions)
    monkeypatch.setattr(recommender, "get_user_interactions", _no_interactions)
    monkeypatch.setattr(recommender, "get_user_profile_categories", _no_categories)
    monkeypatch.setattr(recommender, "_get_latest_top_stories", _latest_top_stories)
    monkeypatch.setattr(recommender.config, "ENABLE_RECOMMENDATIONS", True)
    # A missing Qdrant handle is swallowed into an empty result before the cold-start check, so seed it.
    monkeypatch.setitem(recommender.state, "qdrant", object())

    return redis_stub


@pytest.fixture
def client(monkeypatch):
    """TestClient with ``require_auth`` bypassed, so the handler is reachable.

    The gate's own behaviour is asserted in
    ``test_no_credentials_is_rejected_before_the_handler_runs``.
    """

    async def _authed(request: Request) -> None:
        request.state.user_id = USER

    monkeypatch.setitem(main.app.dependency_overrides, main.require_auth, _authed)
    return TestClient(main.app)


def test_authenticated_user_with_no_history_gets_cold_start_recommendations(client, env):
    """The real cold-start path: authenticated, zero interactions, latest stories.

    Served by the recommender's ``if not interactions`` branch, so removing or
    emptying that fallback turns this red.
    """
    response = client.get("/recommend/for-you", params={"limit": 5})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cold_start"] is True, "no interaction history must be reported as a cold start"
    assert [a["id"] for a in body["recommendations"]] == [11, 12], (
        "cold start must still surface latest top stories, not an empty list"
    )
    assert body["user_id"] == USER, "the response must be attributed to the authenticated user"
    assert body["cached"] is False


def test_cold_start_response_is_cached_under_the_authenticated_user_id(client, env):
    """A warm second call is served from cache, still attributed to that user.

    The cache key is the only place the user id reaches the cached entry, so a
    response cached under one id and replayed for another would be a real
    cross-user leak.
    """
    assert client.get("/recommend/for-you", params={"limit": 5}).json()["cached"] is False

    second = client.get("/recommend/for-you", params={"limit": 5})

    assert second.status_code == 200, second.text
    body = second.json()
    assert body["cached"] is True
    assert body["user_id"] == USER
    # Built by the endpoint's helper: re-spelling the versioned key format is how writer and reader drift.
    assert main._for_you_cache_key(USER, 5) in env.store, "precondition: the entry is cached"


def test_response_is_never_attributed_to_anonymous(monkeypatch, client, env):
    """No id the auth layer can install produces an ``"anonymous"`` response.

    ``require_auth`` sets one of exactly two things, ``SERVICE_USER_ID`` or a
    real user's id, and both are driven here.
    """
    for user_id in (USER, auth.SERVICE_USER_ID):
        async def _authed(request: Request, uid=user_id) -> None:
            request.state.user_id = uid

        monkeypatch.setitem(main.app.dependency_overrides, main.require_auth, _authed)
        client.app.dependency_overrides[main.require_auth] = _authed

        body = client.get("/recommend/for-you", params={"limit": 5}).json()

        assert body["user_id"] == user_id, f"response must be attributed to {user_id!r}"
        assert body["user_id"] != "anonymous", "there is no anonymous for-you response"
        assert [a["id"] for a in body["recommendations"]] == [11, 12]


def test_no_credentials_is_rejected_before_the_handler_runs(monkeypatch, env):
    """An anonymous caller gets a 401 before the handler runs.

    The route is auth-gated, so there is no anonymous response to return and no
    cold-start state to compute for one.
    """
    monkeypatch.delitem(main.app.dependency_overrides, main.require_auth, raising=False)
    client = TestClient(main.app)

    response = client.get("/recommend/for-you", params={"limit": 5})

    assert response.status_code == 401, response.text
