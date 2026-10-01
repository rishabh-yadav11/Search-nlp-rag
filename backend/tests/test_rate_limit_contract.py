"""Behavioural contract for the Redis-backed rate limiter's window bookkeeping.

The limiter's safety rests on two arguments to a single ``SET``: ``NX`` so the
window is not re-armed on every hit, and ``EX`` so the counter is reclaimed.
Both are invisible to a permissive double that accepts ``**kwargs``, so these
cases assert on OBSERVABLE behaviour instead.

Every test but one installs the fake via ``auth._rate_client`` and asserts
``fake.violations == []``: ``_consume_counter`` wraps the Redis exchange in
``except Exception`` and either fails closed or degrades to the in-process
fallback, so a double that merely raised would keep the suite green while
proving nothing. A violation is data, and it is asserted on.
"""

import asyncio

import pytest
from fastapi import HTTPException
from rate_limit_fake import RateLimitRedisFake

from app import auth
from app.config import config

WINDOW = 60
KEY = "auth:rl:login:203.0.113.7"


@pytest.fixture
def fake(monkeypatch):
    # A frozen clock, so TTL assertions are exact instead of hiding real drift.
    f = RateLimitRedisFake(clock=lambda: 1000.0)
    monkeypatch.setattr(auth, "_rate_client", f)
    return f


def _consume(fake, key=KEY, limit=2, window=WINDOW, fail_closed=True):
    return auth._consume_counter(key, limit, window, action="login", fail_closed=fail_closed)


def test_counter_window_is_attached_and_the_key_is_reclaimed(fake):
    """A TTL is not an optimisation: without it the key is immortal.

    Drive the real limiter to exhaustion, move fake time past the window, and
    confirm the SAME key is served again from a fresh count. A limiter that
    stopped passing ``ex=`` leaves the key alive forever.
    """
    for _ in range(2):
        asyncio.run(_consume(fake, limit=2))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_consume(fake, limit=2))
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == str(WINDOW)

    # The counter carries the window as its expiry; it is not a bare integer.
    assert fake.ttl(KEY) == WINDOW
    assert fake.expiry(KEY) == 1000.0 + WINDOW

    fake.advance(WINDOW + 1)
    assert fake.counters.get(KEY) is None, "the window must have reclaimed the key"

    asyncio.run(_consume(fake, limit=2))
    assert fake.counters[KEY] == 1, "the count must restart once the window closes"

    assert fake.violations == []


def test_incrementing_does_not_extend_the_window(fake):
    """A window that slid forward on every hit would be a window that never
    closes, so ``INCR`` must leave the existing expiry alone."""
    asyncio.run(_consume(fake, limit=100))
    opened = fake.expiry(KEY)
    fake.advance(5)
    asyncio.run(_consume(fake, limit=100))
    assert fake.expiry(KEY) == opened, "INCR must not push the window forward"
    assert fake.violations == []


def test_nx_does_not_reset_a_counter_created_earlier_in_the_window(fake):
    """Without ``NX`` the limiter re-arms its own window on every request.

    That pins the count at 1 forever, so the limit is never reached and the
    endpoint is effectively unrated.
    """

    async def seed():
        await fake.set(KEY, 4, nx=True, ex=WINDOW)

    asyncio.run(seed())
    assert fake.counters[KEY] == 4
    assert fake.calls == [("set", KEY, 4, True, WINDOW)], "the seed is the only call so far"

    asyncio.run(_consume(fake, limit=100))
    assert fake.counters[KEY] == 5, "NX must leave the existing count alone"
    assert fake.violations == []

    # And the limiter's own SET really was declined, not silently accepted.
    set_calls = [c for c in fake.calls if c[0] == "set" and c[1] == KEY]
    assert len(set_calls) == 2
    assert set_calls[1] == ("set", KEY, 0, True, WINDOW)


def test_window_is_established_before_the_first_increment(fake):
    """Order is the whole point: ``SET NX EX`` then ``INCR``.

    If the window-establishing write moved after the increment, ``INCR`` would
    create the key first and that key would carry no expiry. The ``NX`` SET that
    follows is then DECLINED, so nothing ever attaches a TTL.
    """
    asyncio.run(_consume(fake, limit=100))
    assert fake.calls == [
        ("set", KEY, 0, True, WINDOW),
        ("incr", KEY),
    ]
    assert fake.violations == []


def test_fake_records_a_counter_written_without_a_ttl():
    """The double's own acceptance clause, tested on the double.

    Production wraps the Redis exchange in ``except Exception``, so a fake that
    raised here would push the request onto the in-process fallback. It records
    instead -- and still performs the write, so the consequence is observable.
    """
    fake = RateLimitRedisFake(clock=lambda: 2000.0)
    asyncio.run(fake.set(KEY, 0, nx=True))

    assert len(fake.violations) == 1
    assert KEY in fake.violations[0]
    assert "no TTL" in fake.violations[0]
    # The write happened anyway; the violation is the signal, not a refusal.
    assert fake.counters[KEY] == 0
    assert fake.ttl(KEY) is None

    # A real TTL is not a violation.
    clean = RateLimitRedisFake()
    asyncio.run(clean.set(KEY, 0, nx=True, ex=WINDOW))
    assert clean.violations == []


def test_in_process_fallback_honours_its_window(monkeypatch):
    """The degraded path is still a limiter, with the same fixed window.

    It is the path every request in a Redis-less deployment actually takes.
    """
    clock = {"t": 1000.0}
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock["t"])

    key = "auth:rl:acct:login:someone@example.com"
    counts = [auth._local_rate_hit(key, WINDOW) for _ in range(3)]
    assert counts == [1, 2, 3]

    clock["t"] += WINDOW - 1
    assert auth._local_rate_hit(key, WINDOW) == 4, "the window must still be open one second before it closes"

    clock["t"] += 2
    assert auth._local_rate_hit(key, WINDOW) == 1, "the count must restart once the window closes"


def test_public_limit_uses_its_own_window_and_fails_closed(monkeypatch):
    """The public surface differs from the auth surface in both knobs, and both
    reach the same ``SET NX EX`` line -- so one fake covers both."""
    fake = RateLimitRedisFake(clock=lambda: 3000.0)
    monkeypatch.setattr(auth, "_rate_client", fake)
    monkeypatch.setattr(auth, "_client_ip", lambda r: "198.51.100.9")

    window = config.PUBLIC_RATE_WINDOW_SECONDS
    request = type("Req", (), {"client": None, "headers": {}})()

    async def hit():
        await auth._check_rate_limit(
            request,
            "search",
            config.PUBLIC_SEARCH_RATE_PER_MIN,
            key_prefix="public:rl",
            window_seconds=window,
            fail_closed=True,
        )

    key = "public:rl:search:198.51.100.9"
    asyncio.run(hit())
    assert fake.calls == [("set", key, 0, True, window), ("incr", key)]
    assert fake.ttl(key) == window
    assert fake.violations == []
