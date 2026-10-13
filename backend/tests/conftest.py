

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_here = Path(__file__).resolve().parent
_backend_root = _here.parent

if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

_DATA_DIR = _here / ".testdata"
_DATA_DIR.mkdir(exist_ok=True)

os.environ.setdefault("CHAT_DB_PATH", str(_DATA_DIR / "chat.db"))
os.environ.setdefault("AUTH_DB_PATH", str(_DATA_DIR / "auth.db"))
os.environ.setdefault("FEED_DB_PATH", str(_DATA_DIR / "feed.db"))
os.environ.setdefault("QUERY_FIX_VOCAB_PATH", str(_DATA_DIR / "query_vocab.json.gz"))


os.environ.setdefault("ENABLE_QUERY_FIX", "false")
os.environ.setdefault("AUTH_TOKEN_PURGE_INTERVAL_SECONDS", "0")
os.environ.setdefault("CHAT_PURGE_INTERVAL_SECONDS", "86400")
os.environ.setdefault("LLM_DAILY_BUDGET_USD", "0")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")
os.environ.setdefault("AUTH_SERVICE_TOKEN", "test-service-token-backend-suite-7f3")
os.environ.setdefault(
    "AUTH_SERVICE_TOKEN_SCOPE", "chat:use,analytics:read,users:read,users:manage"
)
os.environ.setdefault("AUTH_ADMIN_EMAIL", "admin@example.test")
os.environ.setdefault("AUTH_ADMIN_PASSWORD", "admin-pass-2026")




os.environ.setdefault("FX_RATE_API_URL", "http://127.0.0.1:1/fx")
os.environ.setdefault("FX_RATE_REFRESH_SECONDS", "0")



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



os.environ.setdefault("GEMINI_API_KEY", "AIza0123456789AbCdEfGhIjKlMnOpQrStUvWxY")


@pytest.fixture(scope="session")
def app_client():

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

    from app import main

    return main.state["qdrant"]


@pytest.fixture(scope="function")
def fake_redis_factory(app_client):

    import redis.asyncio as aioredis

    return aioredis.from_url


@pytest.fixture(scope="function")
def frozen_clock(monkeypatch):

    import app.query_intent as qi

    frozen = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(qi, "_now", lambda: frozen)
    return frozen


@pytest.fixture(scope="function")
def unique_email():

    import itertools

    counter = itertools.count()
    return f"user-{next(counter)}-{os.getpid()}@example.test"


@pytest.fixture(scope="function")
def admin_headers(app_client):

    import os as _os

    return {"X-Service-Token": _os.environ["AUTH_SERVICE_TOKEN"]}


FROZEN_ISO = "2026-09-15"
FROZEN_DATETIME = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
FROZEN_DAY_AGO_ISO = (FROZEN_DATETIME - timedelta(days=1)).date().isoformat()
