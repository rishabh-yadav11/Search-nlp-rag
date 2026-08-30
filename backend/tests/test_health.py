"""Health probe tests: /health, /live, the Qdrant/models/LLM/Redis checks, and
the /ready + /readyz readiness endpoints (200 vs 503). Redis reachability is
mocked so no real Redis is needed; the module-global ``_redis_client`` is reset
between tests."""

import asyncio

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


# Slack over the close timeout: a correctly bounded teardown finishes near
# _REDIS_CLOSE_TIMEOUT, an unbounded one never finishes and trips this guard.
_HUNG_CLOSE_GUARD = health._REDIS_CLOSE_TIMEOUT + 5.0


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


class _EqualSentinel:
    """Client stand-in whose instances compare equal but are not identical.

    Two distinct instances satisfy ``==`` while failing ``is``, so only an
    identity check can separate them - a guard regressed to value comparison
    cannot."""

    def __init__(self, tag):
        self.tag = tag
        self.closed = 0

    def __eq__(self, other):
        return isinstance(other, _EqualSentinel)

    def __hash__(self):
        return hash(_EqualSentinel)

    async def aclose(self):
        self.closed += 1


def test_drop_redis_client_is_identity_guarded(monkeypatch):
    """Invalidation only clears the client that actually failed: a client
    installed by another caller in the meantime survives.

    The sentinels compare equal while being distinct objects, so swapping the
    ``is`` guard for ``==`` would null the surviving client and fail here."""
    stale = _EqualSentinel("stale")
    current = _EqualSentinel("current")
    assert stale == current and stale is not current
    monkeypatch.setattr(health, "_redis_client", current)

    _run(health._drop_redis_client(stale))
    assert health._redis_client is current
    assert (stale.closed, current.closed) == (0, 0)

    _run(health._drop_redis_client(current))
    assert health._redis_client is None
    # Only the client that was actually dropped gets released.
    assert (stale.closed, current.closed) == (0, 1)


def test_drop_redis_client_closes_the_failing_client(monkeypatch):
    """The dropped client's pool is closed instead of being left to the GC."""
    stale = _EqualSentinel("stale")
    monkeypatch.setattr(health, "_redis_client", stale)

    _run(health._drop_redis_client(stale))

    assert health._redis_client is None
    assert stale.closed == 1


def test_drop_redis_client_close_failure_is_suppressed(monkeypatch):
    """A client that failed its ping may fail to close too; teardown errors
    must not mask the degraded-readiness result."""

    class FailingClose:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1
            raise redis.exceptions.RedisError("socket already gone")

    client = FailingClose()
    monkeypatch.setattr(health, "_redis_client", client)

    _run(health._drop_redis_client(client))

    assert health._redis_client is None
    assert client.closed == 1


def test_close_quietly_lets_cancellation_propagate(monkeypatch):
    """Only ``Exception`` is swallowed: a cancellation raised by the close (or
    by the probe being cancelled) is a ``BaseException`` and must unwind, as the
    docstring states."""

    class CancelledClose:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1
            raise asyncio.CancelledError()

    client = CancelledClose()
    with pytest.raises(asyncio.CancelledError):
        _run(health._close_quietly(client))
    assert client.closed == 1


def test_drop_redis_client_without_close_method(monkeypatch):
    """Doubles (and bare sentinels) with no close method are dropped cleanly."""
    client = object()
    monkeypatch.setattr(health, "_redis_client", client)

    _run(health._drop_redis_client(client))

    assert health._redis_client is None


def test_drop_redis_client_closes_outside_the_init_lock(monkeypatch):
    """The close runs after the critical section: a close that re-enters the
    (non-reentrant) init lock must not deadlock."""
    stale = _EqualSentinel("stale")
    replacement = _EqualSentinel("replacement")
    reentered = []

    async def reentrant_aclose():
        reentered.append("close")
        await health._drop_redis_client(replacement)

    stale.aclose = reentrant_aclose
    monkeypatch.setattr(health, "_redis_client", stale)

    async def scenario():
        # A hang would mean the close ran while the lock was still held.
        await asyncio.wait_for(health._drop_redis_client(stale), timeout=2.0)

    _run(scenario())

    assert reentered == ["close"]
    assert health._redis_client is None


def test_drop_redis_client_hung_close_is_abandoned(monkeypatch):
    """A close that never completes is cancelled at the close timeout, so a
    dying connection cannot stall the probe that is releasing it.

    Without the bound, ``await outcome`` in ``_close_quietly`` never returns and
    the guard below trips."""
    close_started = asyncio.Event()

    class HungClose:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1
            close_started.set()
            await asyncio.Event().wait()  # never set: the teardown hangs

    stale = HungClose()
    monkeypatch.setattr(health, "_redis_client", stale)

    async def scenario():
        started = asyncio.get_running_loop().time()
        await asyncio.wait_for(health._drop_redis_client(stale), timeout=_HUNG_CLOSE_GUARD)
        return asyncio.get_running_loop().time() - started

    elapsed = _run(scenario())

    # The client is still invalidated, and the probe returned on time.
    assert health._redis_client is None
    assert stale.closed == 1
    assert close_started.is_set()
    assert elapsed < _HUNG_CLOSE_GUARD


def test_redis_status_hung_close_still_reports_degraded(monkeypatch):
    """End-to-end: a failed ping whose client hangs on close must not hang the
    readiness probe either. The result is still degraded."""
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    created = []

    class FailingHungRedis:
        def __init__(self):
            self.closed = 0

        async def ping(self):
            raise redis.exceptions.RedisError("redis down")

        async def aclose(self):
            self.closed += 1
            await asyncio.Event().wait()  # never set: the teardown hangs

    def fake_from_url(url, **kwargs):
        client = FailingHungRedis()
        created.append(client)
        return client

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)

    async def scenario():
        return await asyncio.wait_for(health._redis_status(), timeout=_HUNG_CLOSE_GUARD)

    assert _run(scenario()) == (False, "degraded")
    assert health._redis_client is None
    assert len(created) == 1
    assert created[0].closed == 1


def test_redis_status_ping_failure_closes_the_client(monkeypatch):
    """End-to-end: a failed ping drops *and* closes the cached client."""
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    created = []

    class FailingRedis:
        def __init__(self):
            self.closed = 0

        async def ping(self):
            raise redis.exceptions.RedisError("redis down")

        async def aclose(self):
            self.closed += 1

    def fake_from_url(url, **kwargs):
        client = FailingRedis()
        created.append(client)
        return client

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)
    assert _run(health._redis_status()) == (False, "degraded")

    assert health._redis_client is None
    assert len(created) == 1
    assert created[0].closed == 1


def test_redis_status_stale_ping_failure_keeps_replacement(monkeypatch):
    """A slow failing ping must not invalidate a client created after it started.

    The ping runs outside the init lock, so it can resolve after another caller
    already dropped the dead client and reconnected. Nulling ``_redis_client``
    unconditionally discards that healthy replacement and forces another
    reconnect (issue #184)."""
    monkeypatch.setattr(config, "REDIS_URL", "redis://x/0")
    created = []
    stale_ping_started = asyncio.Event()
    release_stale = asyncio.Event()

    class StaleRedis:
        def __init__(self):
            self.closed = 0

        async def ping(self):
            stale_ping_started.set()
            await release_stale.wait()
            raise redis.exceptions.RedisError("connection died")

        async def aclose(self):
            self.closed += 1

    class FreshRedis:
        def __init__(self):
            self.closed = 0

        async def ping(self):
            return True

        async def aclose(self):
            self.closed += 1

    def fake_from_url(url, **kwargs):
        client = FreshRedis() if created else StaleRedis()
        created.append(client)
        return client

    monkeypatch.setattr(health.aioredis, "from_url", fake_from_url)

    async def scenario():
        stale_ping = asyncio.create_task(health._redis_status())
        await stale_ping_started.wait()
        # Second caller's own ping also failed, so it drops the dead client
        # (releasing it) and reconnects while the first ping is still in flight.
        stale = health._redis_client
        await health._drop_redis_client(stale)
        assert await health._redis_status() == (True, "redis")
        replacement = health._redis_client
        release_stale.set()
        return await stale_ping, replacement

    (ok, mode), replacement = _run(scenario())
    assert (ok, mode) == (False, "degraded")
    assert health._redis_client is replacement
    assert len(created) == 2
    # The healthy replacement is reused, so no third reconnect happens.
    assert _run(health._redis_status()) == (True, "redis")
    assert len(created) == 2
    # Only the client that failed got released; the live replacement is open.
    assert created[0].closed == 1
    assert replacement.closed == 0


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
