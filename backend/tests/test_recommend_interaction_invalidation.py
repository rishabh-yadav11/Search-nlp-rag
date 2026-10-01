"""Regression tests for the /recommend/interaction cache invalidation.

Invalidation must cost a bounded number of key names: a ``SCAN``-based prefix
delete walks every key in the database and only then applies ``MATCH``, so the
endpoint derives the (bounded, knowable) key set and ``DEL``s it directly.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app import main
from app.redis_cache import HybridCache

USER = "user-1"
OTHER_USER = "user-2"


def _for_you_key(user_id: str, limit: int) -> str:
    return main._for_you_cache_key(user_id, limit)


FOR_YOU_KEYS = [
    _for_you_key(USER, n) for n in range(main.FOR_YOU_MIN_LIMIT, main.FOR_YOU_MAX_LIMIT + 1)
]
# Unrelated cache traffic standing in for the rest of the database.
FOREIGN_KEYS = [f"search:q{i}" for i in range(500)]


class _SpyRedis:
    """Redis stand-in that records commands and how many keys a keyspace-wide
    operation had to walk past.

    ``scan_iter`` returns a real async generator, so a reintroduced prefix
    delete runs for real against this store: ``scans``/``scanned_keys`` go up
    and the tests fail, rather than the fake quietly never being called.
    """

    def __init__(self, store):
        # Real Redis hands back serialized values, so the store holds strings and HybridCache does its own json.loads.
        self.store = {key: json.dumps(value) for key, value in store.items()}
        self.scans = 0
        self.scanned_keys = 0
        self.deleted = []
        self.commands = []

    async def get(self, key):
        self.commands.append("get")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.commands.append("set")
        # ``value`` arrives already JSON-encoded; re-encoding here would
        # double-encode, and a later get() would hand back a raw string.
        self.store[key] = value
        return True

    async def delete(self, *keys):
        self.commands.append("delete")
        self.deleted.extend(keys)
        for key in keys:
            self.store.pop(key, None)
        return len(keys)

    def scan_iter(self, match=None, count=None):
        self.scans += 1

        async def _walk():
            # SCAN's defining property: MATCH filters keys it already walked, it does not walk fewer of them.
            for key in list(self.store):
                self.scanned_keys += 1
                if match is None or key.startswith(match.rstrip("*")):
                    yield key

        return _walk()

    async def aclose(self):
        self.commands.append("aclose")


@pytest.fixture
def env(monkeypatch):
    """Wire the recommendation endpoints to a real HybridCache over a spy Redis,
    with the interaction/profile Redis calls stubbed out."""
    store = {key: [{"id": 1}] for key in [*FOREIGN_KEYS, *FOR_YOU_KEYS]}
    redis_spy = _SpyRedis(store)

    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=10_000, max_bytes=1 << 24)
    cache._redis = redis_spy
    monkeypatch.setattr(main, "cache", cache)
    interactions = []
    recommend_calls = []

    async def _record_interaction(**kwargs):
        interactions.append(kwargs)

    async def _invalidate_profile(user_id):
        return None

    async def _get_user_interactions(user_id):
        return []

    async def _get_personalized(user_id, limit, exclude_ids):
        recommend_calls.append({"user_id": user_id, "limit": limit})
        return [{"id": 2, "title": f"rec-{user_id}-{limit}"}]

    monkeypatch.setattr(main, "record_interaction", _record_interaction)
    monkeypatch.setattr(main, "invalidate_user_profile", _invalidate_profile)
    monkeypatch.setattr(main, "get_user_interactions", _get_user_interactions)
    monkeypatch.setattr(main, "get_personalized_recommendations", _get_personalized)

    return SimpleNamespace(
        redis=redis_spy, cache=cache, interactions=interactions, recommend_calls=recommend_calls
    )



@pytest.fixture(autouse=True)
def _working_interaction_limiter(monkeypatch):
    """Give the interaction endpoint's rate limiter a working counter store.

    The per-IP and per-account limits fail CLOSED: with no limiter Redis the
    route answers 503 rather than serve an unlimited write path. These tests are
    about cache invalidation, so they must not paper over that behaviour.
    """
    from app import auth

    counters: dict[str, int] = {}

    class _FakeRateRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            counters[key] = counters.get(key, 0) + 1
            return counters[key]

    monkeypatch.setattr(auth, "_rate_client", _FakeRateRedis())
    return counters


@pytest.fixture
def client(monkeypatch, env):
    """TestClient authenticated as USER."""

    # Annotated: an unannotated ``request`` would be read as a *query* parameter and every request would 422.
    async def _authed(request: Request) -> None:
        request.state.user_id = USER

    monkeypatch.setitem(main.app.dependency_overrides, main.require_auth, _authed)
    return TestClient(main.app)


def _post_interaction(client, article_id=7):
    return client.post(
        "/recommend/interaction",
        json={"article_id": article_id, "interaction_type": "click"},
    )


def test_interaction_invalidates_for_you_cache_without_scanning_the_keyspace(client, env):
    """The whole point: no keyspace walk on a request path.

    Asserted structurally rather than by timing -- the spy counts SCAN
    invocations and the keys they were made to walk, so a request must cost a
    bounded number of key names however many unrelated keys the database holds.
    """
    assert env.redis.scans == 0 and env.redis.scanned_keys == 0, "precondition: nothing scanned yet"
    assert len(FOREIGN_KEYS) > 10 * len(FOR_YOU_KEYS), "the store must be big enough for a scan to hurt"

    response = _post_interaction(client)

    assert response.status_code == 200, response.text
    assert env.redis.scans == 0, "a request must not SCAN the keyspace"
    assert env.redis.scanned_keys == 0, "a request must not walk a single key it did not name"
    # The at-most-20 per-user entries, deleted by name in one round trip.
    assert len(env.redis.deleted) <= len(FOR_YOU_KEYS)
    assert env.redis.commands.count("delete") == 1, "the key set must be deleted in a single command"
    for key in FOR_YOU_KEYS:
        assert key not in env.redis.store, f"{key} should have been invalidated"
    for key in FOREIGN_KEYS:
        assert key in env.redis.store, f"{key} is unrelated and must survive"


def test_the_cache_exposes_no_scan_based_delete(client, env):
    """There is no keyspace-scan delete left for a request path to reach for:
    the primitive is gone, not merely unused, which is what stops the
    regression from creeping back in as a convenient prefix invalidation."""
    assert not hasattr(env.cache, "delete_prefix"), "the SCAN-based helper must not come back"
    assert hasattr(env.cache, "delete_keys"), "the derived-key-set delete must be the primitive"

    assert _post_interaction(client).status_code == 200

    assert env.redis.scans == 0
    assert env.redis.scanned_keys == 0


def test_interaction_leaves_another_users_for_you_cache_alone(client, env):
    other_key = _for_you_key(OTHER_USER, 5)
    env.redis.store[other_key] = json.dumps([{"id": 3}])

    assert _post_interaction(client).status_code == 200

    assert other_key in env.redis.store, "invalidating one user must not evict another user's cache"
    assert other_key not in env.redis.deleted


def test_for_you_recomputes_after_an_interaction(client, env):
    """Invalidation must still be *correct*: a stale hit here is a silent personalization bug."""
    before = client.get("/recommend/for-you", params={"limit": 5})
    assert before.status_code == 200, before.text
    assert before.json()["cached"] is True, "precondition: a warm entry is served from cache"
    assert env.recommend_calls == []

    assert _post_interaction(client).status_code == 200

    after = client.get("/recommend/for-you", params={"limit": 5})
    assert after.status_code == 200, after.text
    assert after.json()["cached"] is False, "the interaction must have dropped the cached entry"
    assert len(env.recommend_calls) == 1, "the recommendation must actually be recomputed"

    # The recomputed entry is cached again, so invalidation did not degrade into a permanently missing cache.
    rewarmed = client.get("/recommend/for-you", params={"limit": 5})
    assert rewarmed.json()["cached"] is True
    assert len(env.recommend_calls) == 1


def test_every_writable_for_you_limit_is_invalidated(client, env):
    """The invalidation set must cover the WRITABLE band, proved from the
    request side and never from the constants.

    Building expectations from ``FOR_YOU_MIN_LIMIT``/``FOR_YOU_MAX_LIMIT`` would
    restate ``_for_you_cache_keys`` instead of testing it: a widened Query with
    an unchanged invalidation range would still pass. So this drives real
    requests, discovers the accepted band, and requires every key the endpoint
    actually wrote to be gone -- matched by string prefix, not list membership.
    """
    prefix = _for_you_key(USER, 0).rsplit(":", 1)[0] + ":"
    env.redis.store = {k: v for k, v in env.redis.store.items() if not k.startswith(prefix)}

    # Probe well past the current bound so a widened Query is caught rather than silently truncated.
    accepted, rejected = [], []
    for limit in range(1, 121):
        response = client.get("/recommend/for-you", params={"limit": limit})
        (accepted if response.status_code == 200 else rejected).append(limit)

    written = [key for key in env.redis.store if key.startswith(prefix)]

    assert _post_interaction(client).status_code == 200

    # The property under test, asserted first and independently of the band arithmetic below.
    survivors = [key for key in env.redis.store if key.startswith(prefix)]
    assert survivors == [], (
        f"{len(survivors)} stale for-you entries survived the interaction "
        f"(accepted limits {min(accepted)}..{max(accepted)}): {survivors[:5]}"
    )

    # Supporting detail: the band is contiguous from 1 and ends where the invalidation range does.
    assert accepted, "the endpoint rejected every limit probed"
    assert min(accepted) == 1 and accepted == list(range(1, max(accepted) + 1)), (
        f"the accepted band must be contiguous and start at 1: {accepted[:5]}..{accepted[-5:]}"
    )
    assert rejected and max(accepted) + 1 == min(rejected), (
        f"the accepted band must stop where the bound is: max {max(accepted)}, "
        f"first rejected {min(rejected) if rejected else None}"
    )
    assert max(accepted) == main.FOR_YOU_MAX_LIMIT, (
        f"the endpoint accepts limit {max(accepted)} but the invalidation range stops at "
        f"{main.FOR_YOU_MAX_LIMIT}"
    )
    assert len(written) == max(accepted), "one cache entry per accepted limit was expected"


def test_interaction_purges_the_in_process_fallback_when_redis_is_down(client, env):
    """The degraded path is the one that can still serve stale personalization.

    ``delete_keys`` drops the in-process entries *before* talking to Redis so an
    outage cannot keep handing back pre-interaction recommendations from the
    fallback tier.
    """
    unreachable = ConnectionError("redis unreachable")

    class _DeadRedis:
        async def get(self, key):
            raise unreachable

        async def set(self, key, value, ex=None):
            raise unreachable

        async def delete(self, *keys):
            raise unreachable

        async def aclose(self):
            return None

    env.cache._redis = _DeadRedis()
    own = _for_you_key(USER, 7)
    unrelated = _for_you_key(OTHER_USER, 7)

    async def scenario():
        # With Redis down both of these land in the in-process tier.
        await env.cache.set(own, [{"id": 9}])
        await env.cache.set(unrelated, [{"id": 8}])
        assert await env.cache.get(own) == [{"id": 9}], "precondition: the fallback serves it"
        assert own in env.cache._mem
        return env.cache._mem_bytes

    before = asyncio.run(scenario())
    assert before > 0

    assert _post_interaction(client).status_code == 200

    async def after():
        assert await env.cache.get(own) is None, (
            "the in-process fallback must not keep serving a pre-interaction recommendation"
        )
        assert await env.cache.get(unrelated) == [{"id": 8}], "another user's entry must survive"

    asyncio.run(after())
    assert own not in env.cache._mem
    assert env.cache._mem_bytes == sum(entry[2] for entry in env.cache._mem.values()), (
        "purging the fallback must return its bytes to the budget"
    )
    assert env.cache._mem_bytes < before
