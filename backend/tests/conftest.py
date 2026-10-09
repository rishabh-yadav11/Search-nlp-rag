"""Shared fixtures for the backend test suite.

Everything a test needs lives here so the test modules stay small and
declarative:

* environment is pinned BEFORE any ``app.*`` import (``app.config`` reads
  ``os.getenv`` at import time) — rate limits disabled, data files redirected
  into a scratch dir, a usable Gemini key configured;
* the four external clients are replaced with fakes before the app's lifespan
  runs: ``AsyncQdrantClient`` and ``AsyncOpenAI`` and the three model classes
  on the ``app.main`` module, plus ``redis.asyncio.from_url`` (every module —
  cache, health, auth limiter, analytics, cost_budget, user_profile — builds
  its Redis client through that one seam);
* a session-scoped ``TestClient`` runs the real FastAPI app with its real
  lifespan, so middleware (TrustedHost, CORS, RequestId), routers and stores
  are exercised for real.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Environment: must be complete before any ``import app.config`` anywhere.
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve().parent
_backend_root = _here.parent
# Make ``app`` importable regardless of how pytest is invoked.
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

_DATA_DIR = _here / ".testdata"
_DATA_DIR.mkdir(exist_ok=True)

os.environ.setdefault("CHAT_DB_PATH", str(_DATA_DIR / "chat.db"))
os.environ.setdefault("AUTH_DB_PATH", str(_DATA_DIR / "auth.db"))
os.environ.setdefault("QUERY_FIX_VOCAB_PATH", str(_DATA_DIR / "query_vocab.json.gz"))

# Determinism switches (documented knobs, all optional):
os.environ.setdefault("ENABLE_QUERY_FIX", "false")          # no typo correction
os.environ.setdefault("AUTH_TOKEN_PURGE_INTERVAL_SECONDS", "0")  # no background purge
os.environ.setdefault("CHAT_PURGE_INTERVAL_SECONDS", "86400")   # retention loop sleeps
os.environ.setdefault("LLM_DAILY_BUDGET_USD", "0")          # cost budget disabled
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")        # cookie travels over http
os.environ.setdefault("AUTH_SERVICE_TOKEN", "test-service-token-backend-suite-7f3")
os.environ.setdefault(
    "AUTH_SERVICE_TOKEN_SCOPE", "chat:use,analytics:read,users:read,users:manage"
)
os.environ.setdefault("AUTH_ADMIN_EMAIL", "admin@example.test")
os.environ.setdefault("AUTH_ADMIN_PASSWORD", "admin-pass-2026")  # letter + digit, >= 8

# The fx_rate loop must not hit the real network during the suite: point it at a
# loopback URL that fails fast, so it logs a fetch failure and keeps the
# configured fallback rate exactly as a real outage would.
os.environ.setdefault("FX_RATE_API_URL", "http://127.0.0.1:1/fx")
os.environ.setdefault("FX_RATE_REFRESH_SECONDS", "0")  # no background refresh

# Rate limits: disabled (0) so a full test session can never trip a 429 and
# never depends on wall-clock windows.
for _env in (
    "AUTH_SIGNUP_RATE_PER_MIN",
    "AUTH_LOGIN_RATE_PER_MIN",
    "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN",
    "PUBLIC_SEARCH_RATE_PER_MIN",
    "PUBLIC_FACETS_RATE_PER_MIN",
    "PUBLIC_CLICK_RATE_PER_MIN",
    "PUBLIC_READY_RATE_PER_MIN",
    "PUBLIC_INTERACTION_RATE_PER_MIN",
    "INTERACTION_USER_RATE_PER_MIN",
    "CLICK_BOOST_MIN_CLICKS",
    "CLICK_BOOST_MIN_ARTICLE_CLICKS",
):
    os.environ.setdefault(_env, "0")

# A key that classifies as "ok" for readiness: AIza + 35 mixed chars, >= 12
# distinct. It is never sent anywhere: AsyncOpenAI is faked.
os.environ.setdefault("GEMINI_API_KEY", "AIza0123456789AbCdEfGhIjKlMnOpQrStUvWxY")


@pytest.fixture(scope="session")
def app_client():
    """A TestClient over the real app whose external clients are all fakes.

    Session-scoped: the lifespan (store connections, fake model/qdrant/llm
    construction, background loops) runs exactly once, and the SQLite stores
    live in the scratch data dir seeded clean at session start.
    """
    import redis.asyncio as aioredis
    from fakes import (
        FakeDenseEncoder,
        FakeLLMClient,
        FakeQdrant,
        FakeRedisFactory,
        FakeReranker,
        FakeSparseModel,
        seed_articles,
    )
    from starlette.testclient import TestClient

    from app import main

    # Fresh scratch data (an existing DB from a previous run would append users
    # and chats; delete before the lifespan opens them).
    for old in _DATA_DIR.glob("*.db"):
        old.unlink(missing_ok=True)

    fake_qdrant = FakeQdrant(seed_articles())
    redis_factory = FakeRedisFactory()

    original_from_url = aioredis.from_url
    original_async_openai = main.AsyncOpenAI
    original_qdrant = main.AsyncQdrantClient
    original_dense = main.DenseEncoder
    original_sparse = main.SparseTextEmbedding
    original_reranker = main.Reranker

    aioredis.from_url = redis_factory
    main.AsyncOpenAI = FakeLLMClient
    main.AsyncQdrantClient = lambda **kw: fake_qdrant
    main.DenseEncoder = FakeDenseEncoder
    main.SparseTextEmbedding = FakeSparseModel
    main.Reranker = FakeReranker

    try:
        # client=("127.0.0.1", ...) makes every request report a direct loopback
        # peer, which is exactly what /ready/deep requires (it 403s any
        # non-loopback caller). It does not weaken any other endpoint: rate
        # limits are disabled and no test sends X-Forwarded-For.
        with TestClient(main.app, client=("127.0.0.1", 54321)) as client:
            yield client
    finally:
        aioredis.from_url = original_from_url
        main.AsyncOpenAI = original_async_openai
        main.AsyncQdrantClient = original_qdrant
        main.DenseEncoder = original_dense
        main.SparseTextEmbedding = original_sparse
        main.Reranker = original_reranker


@pytest.fixture(scope="function")
def fake_qdrant(app_client):
    """The in-memory Qdrant currently installed on ``app.main.state``.

    Tests read/write ``articles`` on this to check filters reached the store.
    """
    from app import main

    return main.state["qdrant"]


@pytest.fixture(scope="function")
def fake_redis_factory(app_client):
    """The FakeRedisFactory installed at ``redis.asyncio.from_url``."""
    import redis.asyncio as aioredis

    return aioredis.from_url


@pytest.fixture(scope="function")
def frozen_clock(monkeypatch):
    """Freeze ``query_intent._now`` to one deterministic instant.

    ``_today`` / ``_current_year`` are derived from ``_now`` (via the IST
    tz conversion), so freezing the seam freezes every date-intent resolver.
    """
    import app.query_intent as qi

    frozen = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(qi, "_now", lambda: frozen)
    return frozen


@pytest.fixture(scope="function")
def unique_email():
    """Deterministic-ish unique address so signup tests never collide."""
    import itertools

    counter = itertools.count()
    return f"user-{next(counter)}-{os.getpid()}@example.test"


@pytest.fixture(scope="function")
def admin_headers(app_client):
    """X-Service-Token header scoped for every admin surface."""
    import os as _os

    return {"X-Service-Token": _os.environ["AUTH_SERVICE_TOKEN"]}


FROZEN_ISO = "2026-09-15"
FROZEN_DATETIME = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
FROZEN_DAY_AGO_ISO = (FROZEN_DATETIME - timedelta(days=1)).date().isoformat()
