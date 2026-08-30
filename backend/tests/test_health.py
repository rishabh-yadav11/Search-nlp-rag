"""Health probe tests: /health, /live, the Qdrant/models/LLM/Redis checks, and
the /ready + /readyz readiness endpoints (200 vs 503). Redis reachability is
mocked so no real Redis is needed; the module-global ``_redis_client`` is reset
between tests."""

import asyncio
import importlib

import pytest
import redis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import health
from app.config import config


def _run(coro):
    return asyncio.run(coro)


def _async(result):
    async def wrapper(*args, **kwargs):
        return result

    return wrapper


async def _concurrent_status(count: int) -> list[tuple[bool, str]]:
    """Run `count` _redis_status calls concurrently (gather needs a live loop)."""
    return await asyncio.gather(*(health._redis_status() for _ in range(count)))


def _recording_lock(created: list) -> type[asyncio.Lock]:
    """Build an asyncio.Lock subclass that records every instance in `created`
    and counts how many callers are queued on it.

    Patching the *class* (rather than the module's lock attribute) is what
    separates these tests from the old spy tests: the lock object under test is
    still the one the module itself owns, so a caller that builds its own lock
    is visible as an extra entry in `created`. acquire() yields once before
    locking so every caller is forced to queue instead of running straight
    through the critical section.
    """

    class RecordingLock(asyncio.Lock):
        def __init__(self):
            super().__init__()
            self.acquires = 0
            self.pending = 0
            self.max_pending = 0
            created.append(self)

        async def acquire(self):
            self.acquires += 1
            self.pending += 1
            self.max_pending = max(self.max_pending, self.pending)
            try:
                await asyncio.sleep(0)  # force the caller to queue on the lock
                return await super().acquire()
            finally:
                self.pending -= 1

    return RecordingLock


def _reload_health() -> None:
    """Restore app.health to its pristine import-time state.

    The regression under test is a property of the module's *initial* state:
    the init lock has to exist before the first caller arrives. Once any caller
    has run, the lazy variant is indistinguishable from the eager one, so these
    tests must not inherit whatever an earlier test left behind. monkeypatch
    cannot rewind a module's globals; a real reload can.
    """
    importlib.reload(health)


class FakeRedis:
    """Redis client stub whose ping always succeeds immediately."""

    async def ping(self):
        return True

    async def aclose(self):
        return None


class SlowPingRedis(FakeRedis):
    """Redis client stub whose ping is slow enough to overlap concurrent callers."""

    async def ping(self):
        await asyncio.sleep(0.02)
        return True


@pytest.fixture(autouse=True)
def _reset_redis_client(monkeypatch):
    monkeypatch.setattr(health, "_redis_client", None)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(health.router)
    tc = TestClient(app)
    try:
        yield tc
    finally:
        tc.close()


# --- /health, /live ---


def test_health_endpoint(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_live_endpoint(client):
    assert client.get("/live").json() == {"status": "ok"}


# --- _qdrant_ok ---


def test_qdrant_ok_client_absent():
    assert _run(health._qdrant_ok({})) is False
    assert _run(health._qdrant_ok({"qdrant": None})) is False


def test_qdrant_ok_success():
    class FakeClient:
        async def collection_exists(self, name):
            return True

    assert _run(health._qdrant_ok({"qdrant": FakeClient()})) is True


def test_qdrant_ok_call_failing(monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException

    class FakeClient:
        async def collection_exists(self, name):
            raise ResponseHandlingException(RuntimeError("qdrant down"))


    assert _run(health._qdrant_ok({"qdrant": FakeClient()})) is False


def test_qdrant_ok_times_out(monkeypatch):
    async def fake_wait_for(coro, timeout):
        try:
            await coro
        finally:
            raise TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    client = type("C", (), {"collection_exists": _async(True)})()
    assert _run(health._qdrant_ok({"qdrant": client})) is False


# --- _models_ok ---


def test_models_ok_all_present():
    full = {"model": object(), "sparse_model": object(), "reranker": object()}
    assert health._models_ok(full) is True


@pytest.mark.parametrize("missing", ["model", "sparse_model", "reranker"])
def test_models_ok_any_missing(missing):
    state = {"model": object(), "sparse_model": object(), "reranker": object()}
    state[missing] = None
    assert health._models_ok(state) is False
    assert health._models_ok({}) is False


# --- _llm_ok ---


def test_llm_ok(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "sk-test")
    assert health._llm_ok() is True
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    assert health._llm_ok() is False


# --- _redis_status ---


def test_redis_status_no_url_uses_memory(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "")
    assert _run(health._redis_status()) == (True, "memory")


def test_redis_status_ping_ok(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:6379/0")
    calls = []

    class FakeRedis:
        async def ping(self):
            calls.append("ping")
            return True

    def fake_from_url(url, **kwargs):
        calls.append(("from_url", url, kwargs))
        return FakeRedis()

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)
    assert _run(health._redis_status()) == (True, "redis")
    tag, url, kwargs = calls[0]
    assert tag == "from_url"
    assert url == "redis://localhost:6379/0"
    assert kwargs["decode_responses"] is True


def test_redis_status_client_reused_between_calls(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    from_url_calls = []

    class FakeRedis:
        async def ping(self):
            return True

    def fake_from_url(url, **kwargs):
        from_url_calls.append(url)
        return FakeRedis()

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)
    _run(health._redis_status())
    _run(health._redis_status())
    assert len(from_url_calls) == 1


def test_redis_status_client_reused_between_concurrent_calls(monkeypatch):
    """Three concurrent first-callers must all queue on the module's one
    pre-existing lock and initialize a single Redis client.

    Regression: with the lock created lazily the module starts with
    ``_redis_init_lock = None``, so the first caller to arrive has to build the
    lock itself and the lock in use afterwards is not the one that existed
    before the calls — there wasn't one. The client count alone cannot detect
    this (the lazy check-then-assign has no await between the two, so late
    callers still share the first lock and rebuild nothing); the lock's
    identity and creation phase can. Reloading first, and patching
    asyncio.Lock rather than the module attribute, is what exposes them.
    """
    created: list[asyncio.Lock] = []
    monkeypatch.setattr(health.asyncio, "Lock", _recording_lock(created))
    _reload_health()  # the module-level lock (if any) is built through the spy
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    from_url_calls = []

    def fake_from_url(url, **kwargs):
        from_url_calls.append(url)
        return SlowPingRedis()

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)

    lock_before = health._redis_init_lock
    results = _run(_concurrent_status(3))

    assert results == [(True, "redis")] * 3
    assert len(from_url_calls) == 1  # one client for all three callers
    assert health._redis_init_lock is lock_before  # never rebuilt by a caller
    assert created == [lock_before]  # exactly one lock, built at import
    assert lock_before.max_pending == 3  # all three queued on that one lock


def test_redis_status_serializes_on_shared_lock(monkeypatch):
    """While the module's lock is held by one caller, no other caller may
    initialize a client — and the lock they queue on is the module's own, not
    one they built on arrival."""
    created: list[asyncio.Lock] = []
    monkeypatch.setattr(health.asyncio, "Lock", _recording_lock(created))
    _reload_health()
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    from_url_calls = []

    def fake_from_url(url, **kwargs):
        from_url_calls.append(url)
        return FakeRedis()

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)

    lock = health._redis_init_lock
    assert created == [lock]  # built once at import, before any caller arrived

    async def scenario():
        await lock.acquire()  # simulate another caller owning the critical section
        pending = asyncio.gather(health._redis_status(), health._redis_status())
        await asyncio.sleep(0.02)
        assert from_url_calls == []  # both callers are queued, none initialized
        assert lock.pending == 2
        lock.release()
        return await pending

    assert _run(scenario()) == [(True, "redis"), (True, "redis")]
    assert len(from_url_calls) == 1
    assert lock.max_pending == 2
    assert health._redis_init_lock is lock


def test_close_redis_keeps_the_shared_init_lock(monkeypatch):
    """close_redis must not drop the init lock: resetting it to None re-opens
    the creation window, so the callers that arrive next build a fresh lock
    instead of the one an in-flight caller is already using."""
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    monkeypatch.setattr(health.aioredis, "from_url", lambda url, **kw: FakeRedis())

    _run(health._redis_status())  # both variants now expose a live lock
    lock_before = health._redis_init_lock
    assert lock_before is not None  # precondition: a client was initialized

    _run(health.close_redis())
    assert health._redis_client is None
    assert health._redis_init_lock is lock_before  # the regression assertion

    _run(_concurrent_status(2))
    assert health._redis_init_lock is lock_before  # follow-up callers reuse it


def test_close_redis_during_inflight_status_keeps_the_lock(monkeypatch):
    """close_redis racing an in-flight _redis_status must leave the lock
    identity untouched: callers arriving after the close have to await the same
    lock the in-flight caller used, not a new one."""
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    monkeypatch.setattr(health.aioredis, "from_url", lambda url, **kw: SlowPingRedis())

    async def scenario():
        in_flight = asyncio.ensure_future(health._redis_status())
        await asyncio.sleep(0)  # it released the lock and is now awaiting its ping
        lock_before = health._redis_init_lock
        assert lock_before is not None  # precondition: it initialized a client

        await health.close_redis()
        assert health._redis_init_lock is lock_before  # the regression assertion
        followups = await asyncio.gather(health._redis_status(), health._redis_status())
        assert health._redis_init_lock is lock_before
        return [await in_flight, *followups]

    assert _run(scenario()) == [(True, "redis")] * 3


def test_redis_status_ping_fails_degraded(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")

    class FakeRedis:
        async def ping(self):
            raise redis.exceptions.RedisError("redis down")

    monkeypatch.setattr(health.aioredis, "from_url", lambda url, **kw: FakeRedis())
    # A ping failure is a real Redis error -> degraded (client reset, not ok).
    assert _run(health._redis_status()) == (False, "degraded")


def test_redis_status_ping_times_out_degraded(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")

    class FakeRedis:
        async def ping(self):
            raise TimeoutError()

    monkeypatch.setattr(health.aioredis, "from_url", lambda url, **kw: FakeRedis())
    # A ping timeout -> degraded (client reset, not ok).
    assert _run(health._redis_status()) == (False, "degraded")


# --- _readiness_report ---


def test_readiness_report_ready(monkeypatch):
    monkeypatch.setattr(health, "_qdrant_ok", _async(True))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_redis_status", _async((True, "memory")))
    monkeypatch.setattr(health, "_llm_ok", lambda: True)

    ready, report = _run(health._readiness_report({}))
    assert ready is True
    assert report["ready"] is True
    assert report["checks"]["qdrant"] == {"ok": True}
    assert report["checks"]["models"] == {"ok": True}
    assert report["checks"]["redis"] == {"ok": True, "cache": "memory"}
    assert report["checks"]["llm"] == {"ok": True}


def test_readiness_report_not_ready_qdrant_down(monkeypatch):
    monkeypatch.setattr(health, "_qdrant_ok", _async(False))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_redis_status", _async((True, "degraded")))
    monkeypatch.setattr(health, "_llm_ok", lambda: True)

    ready, report = _run(health._readiness_report({}))
    assert ready is False
    assert report["checks"]["qdrant"] == {"ok": False}
    assert report["checks"]["redis"] == {"ok": True, "cache": "degraded"}


def test_readiness_report_not_ready_models_missing(monkeypatch):
    monkeypatch.setattr(health, "_qdrant_ok", _async(True))
    monkeypatch.setattr(health, "_models_ok", lambda s: False)
    monkeypatch.setattr(health, "_redis_status", _async((True, "memory")))
    monkeypatch.setattr(health, "_llm_ok", lambda: False)

    ready, report = _run(health._readiness_report({}))
    assert ready is False
    assert report["checks"]["models"] == {"ok": False}
    assert report["checks"]["llm"] == {"ok": False}


def test_readiness_report_wires_real_checks(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "sk-test")

    class FakeQdrant:
        async def collection_exists(self, name):
            return True

    state = {
        "model": object(),
        "sparse_model": object(),
        "reranker": object(),
        "qdrant": FakeQdrant(),
    }
    ready, report = _run(health._readiness_report(state))
    assert ready is True
    assert report["checks"]["qdrant"]["ok"] is True
    assert report["checks"]["models"]["ok"] is True
    assert report["checks"]["redis"] == {"ok": True, "cache": "memory"}
    assert report["checks"]["llm"]["ok"] is True


# --- /ready, /readyz ---


def test_ready_200_when_ready(client, monkeypatch):
    monkeypatch.setattr(health, "_readiness_report", _async((True, {"ready": True, "checks": {}})))
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_ready_503_when_not_ready(client, monkeypatch):
    monkeypatch.setattr(health, "_readiness_report", _async((False, {"ready": False, "checks": {}})))
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_readyz_200_when_ready(client, monkeypatch):
    monkeypatch.setattr(health, "_readiness_report", _async((True, {})))
    assert client.get("/readyz").status_code == 200


def test_readyz_503_when_not_ready(client, monkeypatch):
    monkeypatch.setattr(health, "_readiness_report", _async((False, {})))
    assert client.get("/readyz").status_code == 503
