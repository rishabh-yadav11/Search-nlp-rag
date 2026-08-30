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
