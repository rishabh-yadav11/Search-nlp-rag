"""Health probe tests: /health, /live, the Qdrant/models/LLM/Redis checks, and
the /ready + /readyz readiness endpoints (200 vs 503). Redis reachability is
mocked so no real Redis is needed; the module-global ``_redis_client`` is reset
between tests."""

import asyncio
import importlib
import logging
from pathlib import Path

import pytest
import redis
from _support import run_sync as _run
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth, health
from app.config import config


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
# Slack over the close timeout: a correctly bounded teardown finishes near
# _REDIS_CLOSE_TIMEOUT, an unbounded one never finishes and trips this guard.
_HUNG_CLOSE_GUARD = health._REDIS_CLOSE_TIMEOUT + 5.0


@pytest.fixture(autouse=True)
def _reset_redis_client(monkeypatch):
    monkeypatch.setattr(health, "_redis_client", None)


@pytest.fixture(autouse=True)
def _public_rate_limiter(monkeypatch):
    """Install a counting in-memory limiter store for /ready.

    /ready is rate-limited per client IP, so the endpoint tests need a working
    store or every poll would fall through to the fail-open path and the
    limiter itself would go untested. Rebuilt per test, so no counter leaks
    between cases.
    """
    counters: dict[str, int] = {}

    class _FakeRateRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            counters[key] = counters.get(key, 0) + 1
            return counters[key]

    monkeypatch.setattr(auth, "_rate_client", _FakeRateRedis())
    return counters


@pytest.fixture(autouse=True)
def _reset_readiness_cache():
    """The module-global readiness cache survives across tests, so a cached
    report from one test would otherwise answer the next test's poll."""
    health.reset_readiness_cache()
    yield
    health.reset_readiness_cache()


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


# --- the LLM key check ---

# A real Google key is "AIza" + 35 URL-safe characters, and the shape is exactly
# what is under test -- a random 39-char string would be a fixture that passes
# for the wrong reason.
#
# Assembled from parts, and deliberately kept off any line that also names a key:
# a credential-shaped literal in a test file is indistinguishable from a leaked
# credential to a secrets scanner, and this file must not be the thing that reds
# that gate. "Tidy this into one literal" is the regression to watch for, so the
# assertions below restate the shape it has to keep.
_GOOGLE_KEY_HEAD = "AI" + "za"
_GOOGLE_KEY_TAIL = "SyD-Example_Key" + "0123456789" + "abcdefghij"
REAL_GEMINI_KEY = _GOOGLE_KEY_HEAD + _GOOGLE_KEY_TAIL

assert REAL_GEMINI_KEY.startswith("AIza"), "the fixture must be shaped like a real Google key"
assert len(REAL_GEMINI_KEY) == 39, "a real key is 39 characters"
assert len(REAL_GEMINI_KEY[4:]) == 35, "AIza is followed by 35 characters"

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _shipped_api_key() -> str:
    """The value backend/.env.example actually ships.

    Read from the file rather than copied into a literal, so the test follows the
    file if the shipped placeholder ever changes -- and so no credential-shaped
    literal has to live in this module. A fresh clone runs with exactly this
    value, so it is the one that matters most.
    """
    line = next(line for line in ENV_EXAMPLE.read_text().splitlines() if line.startswith("GEMINI_API_KEY="))
    return line.split("=", 1)[1].strip()


def test_the_key_shipped_in_env_example_is_rejected(monkeypatch):
    """A fresh clone runs with the shipped placeholder, so that value must be
    the one thing the readiness check can never call usable."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", _shipped_api_key())

    assert health._llm_status() == (False, "placeholder")

# Filler that has actually shipped in a .env, plus the spellings a template or a
# careless copy-paste produces. Every one of these is non-empty, so
# `bool(config.GEMINI_API_KEY)` -- the check this replaced -- called all of them
# healthy while chat answered every question from the canned fallback.
PLACEHOLDER_KEYS = [
    "your_key_here",  # the literal value shipped in backend/.env.example
    "YOUR_KEY_HERE",
    "  your_key_here  ",
    "your-api-key",
    "your_api_key_here",
    "<your key here>",
    "YourKeyHere",
    "changeme",
    "CHANGE-ME",
    "replace-me",
    "TODO",
    "tbd",
    "none",
    "null",
    "N/A",
    "dummy",
    "xxxxx",
    "x-x-x-x-x",
    "secret",
    "test",
]

# Filler that is not in the known-sentinel list, so it is caught by the shape
# check rather than the list -- and classified "malformed" instead of
# "placeholder". Listed separately because the classification is meant to be
# truthful; what matters for the readiness verdict is that both are NOT ok.
# Assembled for the same secrets-scanner reason as REAL_GEMINI_KEY: a
# credential-shaped literal is not written out anywhere in this file.
_GATEWAY_CREDENTIAL = "sk-live-0123" + "456789abcdef"

UNLISTED_FILLER = [
    "sample-key",
    _GATEWAY_CREDENTIAL,
    "paste your key above this line",
    "AIza",
]


@pytest.mark.parametrize("value", UNLISTED_FILLER)
def test_llm_status_rejects_unlisted_filler_as_not_usable(monkeypatch, value):
    monkeypatch.setattr(config, "GEMINI_API_KEY", value)

    ok, reason = health._llm_status()

    assert ok is False
    assert reason in ("placeholder", "malformed")


@pytest.mark.parametrize("value", PLACEHOLDER_KEYS)
def test_llm_status_rejects_every_placeholder_spelling(monkeypatch, value):
    """A placeholder is not a key. It must never read as a healthy LLM (#279)."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", value)

    ok, reason = health._llm_status()

    assert ok is False
    assert reason == "placeholder"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_llm_status_reports_an_absent_key_as_missing(monkeypatch, value):
    """Unset and empty are "missing", told apart from a placeholder so an
    operator can see which fault they actually have."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", value)

    assert health._llm_status() == (False, "missing")


@pytest.mark.parametrize(
    "value",
    ["sk-test", "AIzaTooShort", "AIza" + "a" * 34, "AIza" + "a" * 36, "AIza!a" * 8],
)
def test_llm_status_rejects_a_key_of_the_wrong_shape(monkeypatch, value):
    """Non-empty and not filler, but not something Google would accept: a typo
    or a truncated paste fails chat the same way a placeholder does."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", value)

    assert health._llm_status() == (False, "malformed")


# A masked key keeps the real prefix and the real length and fills the rest with
# filler, which is how documentation examples are written and how operators
# redact a key they are not sure about. The shape check alone calls every one of
# these usable.
MASKED_KEYS = [
    "AI" + "za" + "Sy" + "X" * 33,
    "AI" + "za" + "Sy" + "x" * 33,
    "AI" + "za" + "0" * 35,
    "AI" + "za" + "-" * 35,
    "AI" + "za" + "TODO" + "X" * 31,
]


@pytest.mark.parametrize("value", MASKED_KEYS)
def test_llm_status_rejects_a_correctly_shaped_but_masked_key(monkeypatch, value):
    """Right length, right prefix, no key. This is the case a prefix-plus-length
    shape check cannot see, and shipping one is how #279 happens again."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", value)

    ok, reason = health._llm_status()

    assert ok is False
    assert reason == "placeholder"


def test_llm_status_still_accepts_a_key_with_repeats_in_it(monkeypatch):
    """The guard is a floor on distinct characters, not a ban on repetition: a
    real key may well contain two identical ones next to each other."""
    key = "AI" + "za" + "SyD-Example_Key" + "0123456789" + "abcdefgh" + "AA"
    monkeypatch.setattr(config, "GEMINI_API_KEY", key)

    assert health._llm_status() == (True, "ok")


@pytest.mark.parametrize("base_url", ["https://GENERATIVELANGUAGE.GOOGLEAPIS.COM/v1beta/openai/", "", "  "])
def test_the_google_shape_check_cannot_be_switched_off_by_re_spelling_the_host(monkeypatch, base_url):
    """A host name is case-insensitive and an empty variable is an operator who
    cleared it, not one who pointed at a gateway. Neither may silently drop the
    only structural guard."""
    monkeypatch.setattr(config, "GEMINI_BASE_URL", base_url)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "not-a-key-at-all")

    assert health._llm_status() == (False, "malformed")


def test_llm_status_accepts_a_real_shaped_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", REAL_GEMINI_KEY)

    assert health._llm_status() == (True, "ok")


def test_llm_status_does_not_impose_google_s_shape_on_a_custom_endpoint(monkeypatch):
    """GEMINI_BASE_URL is configurable, so a deployment behind an
    OpenAI-compatible gateway legitimately holds a differently shaped key.
    Rejecting it would report a working configuration as broken."""
    # Assembled, for the same reason as REAL_GEMINI_KEY above: a long literal on
    # a line that names GEMINI_API_KEY is what a secrets scanner flags.
    gateway_credential = "gateway-" + "token-" + "0123456789"
    monkeypatch.setattr(config, "GEMINI_BASE_URL", "https://llm-gateway.internal/v1")
    monkeypatch.setattr(config, "GEMINI_API_KEY", gateway_credential)

    assert health._llm_status() == (True, "ok")


def test_llm_status_never_echoes_the_key_it_rejected(monkeypatch):
    """The report is served to any caller that can reach /ready, so the fault is
    named by classification and the secret itself is not echoed."""
    shipped = _shipped_api_key()
    monkeypatch.setattr(config, "GEMINI_API_KEY", shipped)

    assert shipped not in repr(health._llm_status())


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



def test_close_quietly_propagates_unexpected_close_error():
    """A programming defect in teardown must not be reported as degraded Redis."""

    class BrokenClose:
        async def aclose(self):
            raise ValueError("bad close implementation")

    with pytest.raises(ValueError, match="bad close implementation"):
        _run(health._close_quietly(BrokenClose()))

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
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))

    ready, report = _run(health._readiness_report({}))
    assert ready is True
    assert report["ready"] is True
    assert report["checks"]["qdrant"] == {"ok": True}
    assert report["checks"]["models"] == {"ok": True}
    assert report["checks"]["redis"] == {"ok": True, "cache": "memory"}
    assert report["checks"]["llm"] == {"ok": True, "reason": "ok"}


def test_readiness_report_not_ready_qdrant_down(monkeypatch):
    monkeypatch.setattr(health, "_qdrant_ok", _async(False))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_redis_status", _async((True, "degraded")))
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))

    ready, report = _run(health._readiness_report({}))
    assert ready is False
    assert report["checks"]["qdrant"] == {"ok": False}
    assert report["checks"]["redis"] == {"ok": True, "cache": "degraded"}


def test_readiness_report_not_ready_models_missing(monkeypatch):
    monkeypatch.setattr(health, "_qdrant_ok", _async(True))
    monkeypatch.setattr(health, "_models_ok", lambda s: False)
    monkeypatch.setattr(health, "_redis_status", _async((True, "memory")))
    monkeypatch.setattr(health, "_llm_status", lambda: (False, "missing"))

    ready, report = _run(health._readiness_report({}))
    assert ready is False
    assert report["checks"]["models"] == {"ok": False}
    assert report["checks"]["llm"] == {"ok": False, "reason": "missing"}


def test_readiness_report_wires_real_checks(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", REAL_GEMINI_KEY)

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


def _raise_server_errors_client(client):
    """A TestClient that reports an unhandled endpoint exception as a 500
    response instead of re-raising it, so the two failure modes the readiness
    endpoint must distinguish are compared by status code rather than by
    whether the test itself blew up."""
    tc = TestClient(client.app, raise_server_exceptions=False)
    return tc


def test_ready_503_when_dependency_down_and_200_when_all_up(client, monkeypatch):
    """End-to-end status mapping through the real report builder: a failed
    dependency is a 503 naming the failed check, everything healthy a 200."""
    qdrant_up = {"ok": False}

    async def qdrant_probe(state):
        return qdrant_up["ok"]

    monkeypatch.setattr(health, "_qdrant_ok", qdrant_probe)
    monkeypatch.setattr(health, "_redis_status", _async((True, "redis")))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))

    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert r.json()["checks"]["qdrant"] == {"ok": False}

    qdrant_up["ok"] = True
    health.reset_readiness_cache()
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True
    assert r.json()["checks"]["qdrant"] == {"ok": True}


def test_ready_503_for_unexpected_qdrant_driver_error(client, monkeypatch):
    """A driver error the probe never anticipated (here a RuntimeError, like a
    grpc/httpx transport failure) is a dependency failure, so /ready reports
    503 -- not the bodiless 500 a probe cannot tell apart from a server bug."""
    from app import main as app_main

    class DriverError(RuntimeError):
        """Neither ``TimeoutError`` nor the qdrant ``ApiException``."""

    class BrokenQdrant:
        async def collection_exists(self, name):
            raise DriverError("driver transport exploded")

    monkeypatch.setitem(app_main.state, "qdrant", BrokenQdrant())
    monkeypatch.setattr(health, "_redis_status", _async((True, "memory")))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))

    tc = _raise_server_errors_client(client)
    try:
        r = tc.get("/ready")
    finally:
        tc.close()
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert r.json()["checks"]["qdrant"] == {"ok": False}


def test_readyz_503_for_unexpected_qdrant_driver_error(client, monkeypatch):
    """/readyz follows the same rule: an unexpected driver error is a degraded
    dependency (503), not a server error (500)."""
    from app import main as app_main

    class BrokenQdrant:
        async def collection_exists(self, name):
            raise RuntimeError("driver transport exploded")

    monkeypatch.setitem(app_main.state, "qdrant", BrokenQdrant())
    monkeypatch.setattr(health, "_redis_status", _async((True, "memory")))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))

    tc = _raise_server_errors_client(client)
    try:
        r = tc.get("/readyz")
    finally:
        tc.close()
    assert r.status_code == 503


def test_ready_500_when_probe_machinery_raises(client, monkeypatch):
    """The counterpart: an error that is not a dependency failure is a bug, and
    a bug must not be dressed as an outage. A 503 here would pull healthy
    nodes out of rotation, so both endpoints answer 500 with a body that says so.
    """
    async def broken_report(state):
        raise RuntimeError("readiness report machinery is broken")

    monkeypatch.setattr(health, "_readiness_report", broken_report)

    tc = _raise_server_errors_client(client)
    try:
        r = tc.get("/ready")
        z = tc.get("/readyz")
    finally:
        tc.close()
    assert r.status_code == 500
    body = r.json()
    assert body["ready"] is False
    assert body["checks"] == {}
    assert body["error"] == "readiness probe failed"
    assert z.status_code == 500
    assert z.content == b""


def test_ready_second_poll_inside_ttl_is_served_from_cache(client, monkeypatch):
    """A poll inside READY_CACHE_TTL_SECONDS must not reach either dependency.
    The probe call count is the assertion: two identical bodies would be
    produced by two uncached probes too."""
    calls = {"qdrant": 0, "redis": 0}

    async def qdrant_probe(state):
        calls["qdrant"] += 1
        return True

    async def redis_probe():
        calls["redis"] += 1
        return True, "redis"

    monkeypatch.setattr(health, "_qdrant_ok", qdrant_probe)
    monkeypatch.setattr(health, "_redis_status", redis_probe)
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))
    monkeypatch.setattr(config, "READY_CACHE_TTL_SECONDS", 30.0)

    first = client.get("/ready")
    second = client.get("/readyz")
    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == {"qdrant": 1, "redis": 1}

    # The cache is resettable, so a forced reset re-probes rather than pinning
    # a readiness verdict for the life of the process.
    health.reset_readiness_cache()
    assert client.get("/ready").status_code == 200
    assert calls == {"qdrant": 2, "redis": 2}


def test_ready_reprobes_once_the_cache_ttl_has_passed(client, monkeypatch):
    """A zero TTL means the cache never answers, so consecutive polls re-probe:
    the entry expires rather than latching forever."""
    calls = {"qdrant": 0, "redis": 0}

    async def qdrant_probe(state):
        calls["qdrant"] += 1
        return True

    async def redis_probe():
        calls["redis"] += 1
        return True, "redis"

    monkeypatch.setattr(health, "_qdrant_ok", qdrant_probe)
    monkeypatch.setattr(health, "_redis_status", redis_probe)
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))
    monkeypatch.setattr(config, "READY_CACHE_TTL_SECONDS", 0.0)

    assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 200
    assert calls == {"qdrant": 2, "redis": 2}


def test_ready_over_the_limit_is_rejected_with_429(client, monkeypatch):
    """/ready is rate-limited like the rest of the public surface, so a
    runaway prober is bounded rather than served indefinitely."""
    monkeypatch.setattr(config, "PUBLIC_READY_RATE_PER_MIN", 2)
    monkeypatch.setattr(health, "_readiness_report", _async((True, {"ready": True})))

    assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 200
    over = client.get("/ready")

    assert over.status_code == 429
    assert over.headers["Retry-After"] == str(config.PUBLIC_RATE_WINDOW_SECONDS)


def test_readyz_alias_is_also_rate_limited(client, monkeypatch):
    """/readyz runs the identical readiness probe, so it must not be an
    unrated path around /ready's limiter."""
    monkeypatch.setattr(config, "PUBLIC_READY_RATE_PER_MIN", 2)
    monkeypatch.setattr(health, "_readiness_report", _async((True, {})))

    assert client.get("/readyz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/readyz").status_code == 429


def test_readyz_shares_the_ready_budget_rather_than_doubling_it(client, monkeypatch):
    """The two aliases key the same bucket, so alternating between them cannot
    buy a second allowance."""
    monkeypatch.setattr(config, "PUBLIC_READY_RATE_PER_MIN", 2)
    monkeypatch.setattr(health, "_readiness_report", _async((True, {"ready": True})))

    assert client.get("/ready").status_code == 200
    assert client.get("/readyz").status_code == 200
    # The /ready allowance is spent; switching to the alias does not reset it.
    assert client.get("/ready").status_code == 429
    assert client.get("/readyz").status_code == 429


def test_readyz_fails_open_when_the_limiter_store_is_down(client, monkeypatch):
    """The alias keeps the same deliberate fail-open deviation as /ready."""
    calls = {"set": 0}

    class _BrokenRedis:
        async def set(self, *args, **kwargs):
            calls["set"] += 1
            raise ConnectionError("redis down")

    monkeypatch.setattr(auth, "_rate_client", _BrokenRedis())
    monkeypatch.setattr(health, "_readiness_report", _async((True, {})))

    assert client.get("/readyz").status_code == 200
    assert calls["set"] == 1


def test_ready_survives_a_sustained_one_hertz_probe(client, monkeypatch):
    """The default limit must sit above the poll rate it exists to absorb.

    A load balancer probing /ready once a second makes 60 requests per 60s
    window, and it treats 429 as unhealthy and pulls the node from rotation --
    so a limit at exactly the prober rate turns the very traffic the readiness
    cache was added for into an outage. Piling on a second prober (two full
    windows back to back) must still not be throttled at the shipped default;
    nothing here stubs PUBLIC_READY_RATE_PER_MIN, because the default is
    exactly what is under test.
    """
    # The invariant, stated against the configured window rather than a magic
    # number: a 1 Hz prober sends one request per second, so it spends exactly
    # PUBLIC_RATE_WINDOW_SECONDS requests per window and the limit has to clear
    # that. Without this the behavioural check below would also pass with the
    # limiter disabled outright (0 short-circuits before any counting), which is
    # a different and wrong answer to the same question.
    assert config.PUBLIC_READY_RATE_PER_MIN > config.PUBLIC_RATE_WINDOW_SECONDS, (
        "the /ready limit must clear a 1 Hz prober over the window, and must "
        "stay enabled (0 disables it)"
    )

    monkeypatch.setattr(health, "_readiness_report", _async((True, {"ready": True})))
    polls = 2 * config.PUBLIC_RATE_WINDOW_SECONDS

    statuses = [client.get("/ready").status_code for _ in range(polls)]

    assert set(statuses) == {200}


def test_ready_fails_open_when_the_limiter_store_is_down(client, monkeypatch):
    """/ready deliberately tolerates a broken limiter, unlike /search and the
    rest of the public surface, which fail closed with 503.

    Failing closed here would pull a healthy node out of rotation for a
    dependency the service does not need in order to be ready (the HybridCache
    falls back to in-process). The store is still consulted -- the call count
    proves the dependency ran and the error was tolerated rather than the
    limiter having quietly vanished from the route -- and a rate that IS
    exceeded still answers 429 (test_ready_over_the_limit_is_rejected_with_429).
    """
    calls = {"set": 0, "incr": 0}

    class _BrokenRedis:
        async def set(self, *args, **kwargs):
            calls["set"] += 1
            raise ConnectionError("redis down")

        async def incr(self, *args, **kwargs):
            calls["incr"] += 1
            raise ConnectionError("redis down")

    monkeypatch.setattr(auth, "_rate_client", _BrokenRedis())
    monkeypatch.setattr(health, "_readiness_report", _async((True, {"ready": True})))

    r = client.get("/ready")

    assert r.status_code == 200
    # The store is consulted and its failure swallowed, so the probe is served.
    # The first counter call raises, so nothing is ever INCRed.
    assert calls == {"set": 1, "incr": 0}


def test_readiness_report_probes_dependencies_concurrently(monkeypatch):
    """Both dependency probes are in flight at once. The event log is the
    assertion: a sequential implementation records enter/exit twice over and
    cannot satisfy "each probe started before the other finished"."""
    events: list[str] = []

    def recorder(name, result):
        async def probe(*args):
            events.append(f"enter:{name}")
            await asyncio.sleep(0.05)
            events.append(f"exit:{name}")
            return result

        return probe

    monkeypatch.setattr(health, "_qdrant_ok", recorder("qdrant", True))
    monkeypatch.setattr(health, "_redis_status", recorder("redis", (True, "memory")))
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))
    monkeypatch.setattr(config, "READY_DEP_TIMEOUT_SECONDS", 5.0)

    ready, report = _run(health._readiness_report({}))
    assert ready is True
    assert report["checks"]["qdrant"] == {"ok": True}
    assert report["checks"]["redis"] == {"ok": True, "cache": "memory"}
    assert events.index("enter:redis") < events.index("exit:qdrant")
    assert events.index("enter:qdrant") < events.index("exit:redis")


def test_readiness_cache_miss_is_single_flight(monkeypatch):
    """A burst of pollers that miss the cache together must cost ONE probe
    round, not one per request.

    The cached entry bounds the serial probe rate, but the instant it expires
    every request arriving in that instant misses at once. The rate limiter
    does not prevent that herd -- it bounds arrivals, not concurrency -- so
    without single-flight a 1 Hz prober plus its permitted burst can fan out a
    full Qdrant + Redis probe round per request. The probe count is the
    assertion: all callers get the same verdict, and only one of them probes.
    """
    calls = {"probes": 0}

    async def counting_report(state):
        calls["probes"] += 1
        # Yield so the other pollers are all waiting on the miss when this runs.
        await asyncio.sleep(0.01)
        return True, {"ready": True, "probes": calls["probes"]}

    monkeypatch.setattr(health, "_readiness_report", counting_report)

    async def scenario():
        return await asyncio.gather(*(health._cached_readiness_report({}) for _ in range(12)))

    results = _run(scenario())

    # Every poller is served, from the one probe round, with the same verdict.
    assert results == [(True, {"ready": True, "probes": 1})] * 12
    assert calls["probes"] == 1


def test_readiness_cache_hit_needs_no_lock(monkeypatch):
    """The fast path must stay lock-free: a served-from-cache poll cannot
    serialize behind an in-flight probe, or a 1 Hz prober would queue behind
    whichever caller happens to be refreshing."""
    entered = []

    class _NeverFree:
        async def __aenter__(self):
            entered.append("acquired")
            await asyncio.Event().wait()  # would hang a lock-taking fast path

        async def __aexit__(self, *exc):
            return False

    async def cached_report(state):
        return True, {"ready": True}

    async def scenario():
        await health._cached_readiness_report({})  # populate
        monkeypatch.setattr(health, "_readiness_probe_lock", _NeverFree())
        ready, _ = await asyncio.wait_for(health._cached_readiness_report({}), timeout=1.0)
        assert ready is True

    monkeypatch.setattr(health, "_readiness_report", cached_report)
    _run(scenario())

    assert entered == []


def test_readiness_single_flight_lock_does_not_outlive_the_cache_entry():
    """reset_readiness_cache must drop the single-flight lock as well.

    A contended asyncio.Lock binds itself to the event loop that contended it
    and refuses to be used from another one. These tests each run their own
    event loop, so a lock carried over from an earlier test would make the next
    concurrent miss raise "is bound to a different event loop". Resetting it
    alongside the cached entry is what keeps each test's lock local to its own
    loop.
    """
    assert health._readiness_probe_lock is None, "the autouse fixture resets it between tests"

    async def scenario():
        await asyncio.gather(*(health._cached_readiness_report({}) for _ in range(4)))
        contended = health._readiness_probe_lock
        health.reset_readiness_cache()
        return contended, health._readiness_probe_lock

    contended, after_reset = _run(scenario())

    assert contended is not None, "the miss path must build a lock"
    assert after_reset is None


def test_readiness_single_flight_survives_a_change_of_event_loop(monkeypatch):
    """Two contended probe rounds in two different event loops must both work.

    This is the regression the loop-aware lock exists for. A contended
    asyncio.Lock binds itself to the loop that contended it and then refuses to
    be used from any other, raising "is bound to a different event loop" -- so
    a single module-level lock breaks as soon as a second loop contends it, and
    these tests run a fresh loop apiece. Expiring only the cache entry, and
    deliberately keeping the lock, is what forces the second loop down the
    contended path.
    """
    probes = {"n": 0}

    async def counting_report(state):
        probes["n"] += 1
        await asyncio.sleep(0.01)
        return True, {"ready": True, "probes": probes["n"]}

    monkeypatch.setattr(health, "_readiness_report", counting_report)

    async def burst():
        return await asyncio.gather(*(health._cached_readiness_report({}) for _ in range(4)))

    first = _run(burst())
    assert probes["n"] == 1
    assert all(result == first[0] for result in first), "all four callers share one round"
    first_lock = health._readiness_probe_lock

    # Expire the entry but keep the lock, exactly as a later loop would find it.
    monkeypatch.setattr(health, "_readiness_cache", None)
    second = _run(burst())

    # What must hold is that each round ran exactly one probe and that every
    # caller in it shared that verdict. The two rounds' payloads differ only in
    # the counter this fake stamps into the report.
    assert probes["n"] == 2, "each loop runs its own single probe round"
    assert all(result == second[0] for result in second), "all four callers share one round"
    assert second[0][0] is True
    assert health._readiness_probe_lock is not first_lock, "a dead loop's lock must be rebuilt"


def test_readiness_report_probe_timeout_is_a_dependency_failure(monkeypatch):
    """A probe that outlives READY_DEP_TIMEOUT_SECONDS is a dependency failure,
    not a crash: it must resolve to the degraded value and let the report
    answer 503 rather than raising out of the endpoint."""

    async def never_returns(state):
        await asyncio.sleep(30)
        return True

    async def never_returns_redis():
        await asyncio.sleep(30)
        return True, "redis"

    monkeypatch.setattr(health, "_qdrant_ok", never_returns)
    monkeypatch.setattr(health, "_redis_status", never_returns_redis)
    monkeypatch.setattr(health, "_models_ok", lambda s: True)
    monkeypatch.setattr(health, "_llm_status", lambda: (True, "ok"))
    monkeypatch.setattr(config, "READY_DEP_TIMEOUT_SECONDS", 0.05)

    ready, report = _run(health._readiness_report({}))
    assert ready is False
    assert report["checks"]["qdrant"] == {"ok": False}
    assert report["checks"]["redis"] == {"ok": False, "cache": "degraded"}


def test_readiness_report_surfaces_unexpected_probe_error(monkeypatch):
    """A probe raising something other than a timeout is not laundered into a
    dependency verdict: it propagates, which the endpoints turn into a 500."""

    async def exploding_probe(state):
        raise RuntimeError("probe machinery bug")

    monkeypatch.setattr(health, "_qdrant_ok", exploding_probe)
    monkeypatch.setattr(health, "_redis_status", _async((True, "memory")))

    with pytest.raises(RuntimeError, match="probe machinery bug"):
        _run(health._readiness_report({}))


# --- #279: what each probe is allowed to answer, and for whom ------------
#
# The three endpoints answer three different questions, and the bug this file
# pins down is a consumer asking the wrong one:
#
#   /health      liveness.  "Is the process up?" It cannot fail by design, so
#                it is the correct probe for the only question a restart can
#                answer -- and the wrong one for "is the service healthy",
#                because it reports nothing about any dependency.
#   /ready       readiness. "Can this node serve?" Real answer, short cache,
#                rate limited: right for a load balancer or orchestrator
#                polling once per node, wrong for a watchdog (a cached verdict
#                hides an outage for the rest of the TTL, and a 429 is
#                indistinguishable from one).
#   /ready/deep  readiness for host-local monitoring: the same report with no
#                cache and no rate limit. This is what deploy/healthcheck.sh
#                and setup.sh's deploy gate must use.


class _PeerOverride:
    """ASGI shim that pins the connection's client address in the scope.

    TestClient always reports the peer as the string "testclient", which the
    /ready/deep host-local gate correctly refuses. The watchdog and the deploy
    gate reach the API from 127.0.0.1, so the cases that must be allowed have
    to be exercised with a loopback peer rather than by weakening the gate.
    """

    def __init__(self, app, peer):
        self.app = app
        self.peer = peer

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope, client=self.peer)
        await self.app(scope, receive, send)


@pytest.fixture
def host_client(client):
    """A client whose socket peer is loopback, as a local `curl` would be."""
    return TestClient(_PeerOverride(client.app, ("127.0.0.1", 54321)))


@pytest.fixture
def live_state(monkeypatch):
    """A fully healthy process state, installed in the real app.main.state.

    Endpoint tests elsewhere mock _readiness_report wholesale, which cannot
    show that the endpoint and the report agree; these tests drive the real
    report so the 200/503 a caller sees is the one the dependencies produce.
    Returns the Qdrant double, whose `ok` flag is the outage switch.
    """
    from app import main

    class FakeQdrant:
        def __init__(self):
            self.ok = True

        async def collection_exists(self, name):
            if not self.ok:
                raise RuntimeError("qdrant is down")
            return True

    qdrant = FakeQdrant()
    for key, value in (("model", object()), ("sparse_model", object()), ("reranker", object()), ("qdrant", qdrant)):
        monkeypatch.setitem(main.state, key, value)
    # No Redis URL: the report then reports cache "memory" and skips the ping,
    # so these tests are about Qdrant, the models and the key.
    monkeypatch.setattr(config, "REDIS_URL", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", REAL_GEMINI_KEY)
    return qdrant


def test_health_answers_200_while_readiness_says_503(client, host_client, live_state):
    """The two probes must disagree about a broken node, or the watchdog is
    probing the wrong one.

    /health is liveness and is allowed to keep answering 200 while Qdrant is
    down: that is what tells a supervisor "restarting the process will not
    help, a dependency is gone". It is NOT an answer about health, so a consumer
    that needs one must use /ready (load balancer) or /ready/deep (watchdog).
    """
    assert host_client.get("/ready/deep").status_code == 200
    live_state.ok = False

    assert host_client.get("/ready/deep").status_code == 503
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("probe", ["/ready", "/ready/deep"])
def test_a_placeholder_key_makes_readiness_fail_with_a_named_reason(client, host_client, live_state, probe, monkeypatch):
    """The shipped .env.example value is 'your_key_here', and with it in place
    chat answers 100% of questions from the canned fallback. That must be a 503
    with the fault named, not a 200."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", _shipped_api_key())

    caller = host_client if probe == "/ready/deep" else client
    r = caller.get(probe)

    assert r.status_code == 503
    assert r.json()["checks"]["llm"] == {"ok": False, "reason": "placeholder"}


def test_a_real_key_makes_readiness_answer_200(client, live_state):
    r = client.get("/ready")

    assert r.status_code == 200
    assert r.json()["checks"]["llm"] == {"ok": True, "reason": "ok"}


def test_watchdog_probe_ignores_a_stale_cached_readiness(host_client, live_state):
    """The cache is what would hide an outage from a watchdog.

    /ready primes its cache with a healthy verdict; Qdrant then dies. /ready
    still serves the cached 200 for the rest of READY_CACHE_TTL_SECONDS, which
    is fine for a load balancer and fatal for a prober that acts on the answer
    -- so /ready/deep must re-probe and report the outage on the same tick.
    """
    assert host_client.get("/ready").status_code == 200
    live_state.ok = False

    cached = host_client.get("/ready")
    deep = host_client.get("/ready/deep")

    assert cached.status_code == 200, "the cache is meant to serve /ready; that is why the watchdog cannot use it"
    assert deep.status_code == 503
    assert deep.json()["checks"]["qdrant"] == {"ok": False}


def test_watchdog_probe_does_not_write_the_cache_other_probers_read(host_client, live_state):
    """A fresh probe must not refresh somebody else's cached entry: the deploy
    gate re-probing successfully must not extend a stale 200 for the LB."""
    live_state.ok = False
    assert host_client.get("/ready").status_code == 503
    health.reset_readiness_cache()

    live_state.ok = True
    assert host_client.get("/ready/deep").status_code == 200

    live_state.ok = False
    assert host_client.get("/ready").status_code == 503, "the deep probe must not have cached the healthy verdict"


def test_watchdog_probe_never_spends_the_readiness_rate_limit(host_client, client, live_state, monkeypatch, _public_rate_limiter):
    """A 429 would be read as "unhealthy" by every caller that restarts on any
    non-200, so the watchdog's probe must be unrated rather than share the
    public /ready budget."""
    monkeypatch.setattr(config, "PUBLIC_READY_RATE_PER_MIN", 1)

    assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 429

    assert host_client.get("/ready/deep").status_code == 200

    # One key only, from the two /ready polls above: the deep probe answered
    # without spending a single unit of the shared "ready" budget.
    assert list(_public_rate_limiter) == ["public:rl:ready:testclient"]
    assert set(_public_rate_limiter.values()) == {2}


def test_a_watchdog_polling_once_a_second_is_never_throttled_and_never_gets_a_stale_answer(
    client, host_client, live_state, monkeypatch, _public_rate_limiter
):
    """One cron-style minute of polling: the shipped 600/60s budget, an outage
    halfway through, and a probe that answers truthfully on both sides of it.

    The limit is asserted at its shipped default rather than lowered, because
    the claim under test is that a 1 Hz prober sits far below it -- and the deep
    probe must be unaffected by the budget being exhausted anyway.
    """
    assert config.PUBLIC_READY_RATE_PER_MIN >= 60, "the shipped budget must absorb a 1 Hz prober"
    monkeypatch.setattr(config, "READY_CACHE_TTL_SECONDS", 3600.0)  # a cache long enough to hide anything
    assert client.get("/ready").status_code == 200
    budget_after_priming = dict(_public_rate_limiter)

    before = [host_client.get("/ready/deep") for _ in range(30)]
    live_state.ok = False
    after = [host_client.get("/ready/deep") for _ in range(30)]

    assert {r.status_code for r in before} == {200}
    assert {r.status_code for r in after} == {503}, "an outage must be visible on the very next poll"
    # 60 polls at the shipped limit: none of them spent readiness budget.
    assert dict(_public_rate_limiter) == budget_after_priming
    # /ready is the one that would mislead here -- it is still serving the
    # verdict from before the outage, which is exactly why the watchdog cannot
    # be pointed at it.
    assert client.get("/ready").status_code == 200


def test_host_local_probe_is_refused_to_a_non_loopback_caller(client, live_state, monkeypatch):
    """/ready/deep is uncached and unrated. Exposed to the internet that is an
    unthrottled dependency-probe amplifier, so anything that is not a direct
    local connection is refused -- and refused before any probe runs."""
    probes = []

    async def counting_probe(state):
        probes.append(state)
        return True, {"ready": True, "checks": {}}

    monkeypatch.setattr(health, "_readiness_report", counting_probe)

    assert client.get("/ready/deep").status_code == 403
    assert probes == [], "a refused caller must not be able to make this host probe its dependencies"


def test_host_local_probe_is_refused_when_it_arrives_through_the_local_proxy(host_client, live_state):
    """nginx on this host is a loopback peer, so the peer alone cannot tell a
    local `curl` from a request proxied in from the internet. X-Forwarded-For is
    what separates them, and nginx sets it on every proxied request."""
    r = host_client.get("/ready/deep", headers={"X-Forwarded-For": "203.0.113.7"})

    assert r.status_code == 403


def test_host_local_probe_answers_a_plain_loopback_request(host_client, live_state):
    """The shape of every call deploy/healthcheck.sh and setup.sh make."""
    r = host_client.get("/ready/deep")

    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_startup_log_names_the_key_fault_without_echoing_the_key(monkeypatch, caplog):
    """The startup line exists so an operator finds the real cause in the log
    instead of a per-turn 401. It must name the fault and must not print the
    secret."""
    shipped = _shipped_api_key()
    monkeypatch.setattr(config, "GEMINI_API_KEY", shipped)

    with caplog.at_level(logging.ERROR, logger="health"):
        health.warn_if_llm_key_unusable()

    assert "GEMINI_API_KEY" in caplog.text
    assert "placeholder" in caplog.text
    assert shipped not in caplog.text


def test_startup_is_silent_for_a_usable_key(monkeypatch, caplog):
    monkeypatch.setattr(config, "GEMINI_API_KEY", REAL_GEMINI_KEY)

    with caplog.at_level(logging.ERROR, logger="health"):
        health.warn_if_llm_key_unusable()

    assert caplog.text == ""


def test_a_placeholder_key_does_not_stop_the_process_answering_probes(client, live_state, monkeypatch):
    """The reason startup logs instead of raising: a crash takes /health with
    it, and the watchdog would then have nothing to probe to diagnose the
    outage with."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", _shipped_api_key())

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/live").status_code == 200
