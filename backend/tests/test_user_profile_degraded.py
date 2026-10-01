"""Log-volume policy for the Redis-backed user profile helpers.

Every helper keeps serving when its backing store is unreachable, so the log
line is the only signal an operator gets: one WARNING per outage, one WARNING
"recovered" when it ends, and a re-armed latch so a later outage is announced
again. Those lines are rate-limited against the clock -- an outage within
:data:`app.degraded.REANNOUNCE_SECONDS` of the previous line is suppressed by
design -- so a second outage has to move the injected clock past that window.
Both lines are WARNING because the root logger keeps Python's default level
under gunicorn/uvicorn, so a quieter record would never reach PM2.
"""
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app import main, user_profile
from app.degraded import REANNOUNCE_SECONDS, DegradedLatch


class _LatchClock:
    """Callable stand-in for ``time.monotonic`` that moves only when told.

    Injecting the clock keeps the tests from sleeping and makes it explicit
    whether two outages are seconds or minutes apart.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _BrokenRedis:
    """Every command raises, so each caller takes its degraded path."""

    def _boom(self, *_args, **_kwargs):
        raise ConnectionError("profile redis unavailable")

    pipeline = _boom
    delete = _boom
    get = _boom
    hgetall = _boom
    scan = _boom
    set = _boom
    zadd = _boom
    zrevrange = _boom


class _WorkingPipe:
    """Accepts the queued commands and resolves the batch."""

    def delete(self, *_args, **_kwargs):
        return 1

    def expire(self, *_args, **_kwargs):
        return True

    def hincrby(self, *_args, **_kwargs):
        return 1

    def hset(self, *_args, **_kwargs):
        return 1

    def set(self, *_args, **_kwargs):
        return True

    def zincrby(self, *_args, **_kwargs):
        # Per-article detail key written alongside the counter hash.
        return 1

    def zadd(self, *_args, **_kwargs):
        return 1

    def zcard(self, *_args, **_kwargs):
        # One distinct article, comfortably under any cap.
        return 1

    def zscore(self, *_args, **_kwargs):
        # None means no prior interaction, so a write mints a new key.
        return None

    async def execute(self):
        return []


class _WorkingRedis:
    """Serves the handful of commands the helpers actually issue."""

    def pipeline(self, *_args, **_kwargs):
        return _WorkingPipe()

    async def delete(self, *_args, **_kwargs):
        return 1

    # ``_ensure_trending_index`` probes with ``exists``; without it a healthy call reads as an outage.
    async def exists(self, *_args, **_kwargs):
        return False

    async def get(self, *_args, **_kwargs):
        return None

    async def hgetall(self, *_args, **_kwargs):
        return {}

    async def scan(self, cursor, **_kwargs):
        return 0, []

    async def set(self, *_args, **_kwargs):
        return True

    async def zrevrange(self, *_args, **_kwargs):
        return []


class _StoredVectorRedis(_WorkingRedis):
    """Working Redis that also serves a cached profile vector.

    Only a stored value counts as a success for ``get_user_profile_vector``: a
    missing key delegates to ``build_user_profile``, which logs on its own
    latch and would leave this one unrecovered.
    """

    async def get(self, *_args, **_kwargs):
        return "[0.1, 0.2, 0.3]"
 
 
class _KnownArticleRedis(_WorkingRedis):
    """Working Redis that also confirms the article exists.

    ``record_interaction`` confirms the article is indexed before writing, and a
    missing confirmation falls through to Qdrant; serving the cached
    confirmation keeps the write path the one these tests exercise.
    """

    async def get(self, *_args, **_kwargs):
        return "1"
 
 
class _RecordingPipe(_WorkingPipe):
    """Pipeline reporting a slot available and a fresh article counter.

    ``_has_interaction_slot`` unpacks two results and ``record_interaction``
    reads ``results[0]`` as the HINCRBY post-value, so a bare empty list cannot
    walk the write path at all; ``(1, None)`` means "under the cap, no prior
    interaction" plus a brand-new counter.
    """

    async def execute(self):
        return [1, None]


class _RecordableRedis(_KnownArticleRedis):
    """Redis that lets a whole interaction through: confirm, slot, write."""

    def pipeline(self, *_args, **_kwargs):
        return _RecordingPipe()


class _BrokenWriteRedis(_RecordableRedis):
    """Lets the interaction be *attempted*, then fails the write itself.

    The article confirmation and the slot check must both succeed to reach the
    write, and both are Redis round trips: failing the shared pipeline instead
    would fail the slot check too and report a different failure.
    """

    def __init__(self):
        self.queued = 0

    def pipeline(self, *_args, **_kwargs):
        self.queued += 1
        return self

    def delete(self, *_args, **_kwargs):
        return 1

    def hincrby(self, *_args, **_kwargs):
        return 1

    def hset(self, *_args, **_kwargs):
        return 1

    def expire(self, *_args, **_kwargs):
        return True

    def zadd(self, *_args, **_kwargs):
        return 1

    async def execute(self):
        # The first pipeline is the slot check, which must pass; the second is the write, which is down.
        if self.queued == 1:
            return [1, None]
        raise ConnectionError("profile redis unavailable")


class _InteractingRedis(_WorkingRedis):
    """Working Redis that reports one stored interaction.

    ``build_user_profile`` returns early -- logging a recovery, never a failure
    -- when there are no interactions, so its failure path is only reachable
    once there is a profile to build.
    """

    async def zrevrange(self, *_args, **_kwargs):
        return [("1", datetime.now(UTC).timestamp())]


class _BrokenQdrant:
    """Qdrant stand-in whose article lookup can be flipped between up and down."""

    def __init__(self):
        self.broken = False
        self.calls = 0

    async def retrieve(self, **_kwargs):
        self.calls += 1
        if self.broken:
            raise ConnectionError("qdrant unreachable")
        article = SimpleNamespace(
            id=1,
            vector={"dense": [0.1, 0.2, 0.3]},
            payload={"industry_names": ["fintech"], "dealtype_names": []},
        )
        return [article]


@pytest.fixture
def profile_logs(caplog):
    """Capture the records these helpers emit on the ``user_profile`` logger.

    No level override: every line of the policy is WARNING, which caplog
    captures by default.
    """
    return caplog


@pytest.fixture
def clock():
    return _LatchClock()


def _fresh_latches(monkeypatch, clock):
    """Rebind every latch on the injected clock, so no test inherits another's.

    The op names come from the module's own mapping, so a newly latched
    helper is covered without editing this file.
    """
    monkeypatch.setattr(
        user_profile,
        "_latches",
        {
            op: DegradedLatch(user_profile.logger, op, now=clock)
            for op in user_profile._latches
        },
    )


def _use_redis(monkeypatch, client):
    monkeypatch.setattr(user_profile, "_redis_client", lambda: client)


def _events(caplog):
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == user_profile.__name__
    ]


def _levels(caplog):
    return [level for level, _ in _events(caplog)]


@pytest.mark.asyncio
async def test_outage_recovers_then_fails_again_logs_both_outages(
    monkeypatch, profile_logs, clock
):
    """A recovered outage must not silence the next one."""
    _fresh_latches(monkeypatch, clock)
    user_id = "user-acceptance"

    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_user_interactions(user_id) == []

    _use_redis(monkeypatch, _WorkingRedis())
    assert await user_profile.get_user_interactions(user_id) == []

    # Two incidents minutes apart, not one flap: the window has to pass.
    clock.advance(REANNOUNCE_SECONDS + 1)
    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_user_interactions(user_id) == []

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to get user interactions" in events[0][1]
    assert "recovered" in events[1][1]
    assert "get_user_interactions" in events[1][1]
    assert "Failed to get user interactions" in events[2][1]


@pytest.mark.asyncio
async def test_sustained_outage_logs_one_warning(monkeypatch, profile_logs, clock):
    """Five failures in one run are one line, not five.

    The clock never moves, so no re-announce window can open inside this run.
    """
    _fresh_latches(monkeypatch, clock)
    _use_redis(monkeypatch, _BrokenRedis())

    for _ in range(5):
        await user_profile.get_user_interactions("user-sustained-outage")

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING]
    assert "Failed to get user interactions" in events[0][1]


@pytest.mark.asyncio
async def test_healthy_requests_log_nothing(monkeypatch, profile_logs, clock):
    """A dependency that was never down must not announce a recovery."""
    _fresh_latches(monkeypatch, clock)
    _use_redis(monkeypatch, _WorkingRedis())

    for _ in range(3):
        assert await user_profile.get_user_interactions("user-always-healthy") == []
    assert await user_profile.invalidate_user_profile("user-always-healthy") is None

    assert _events(profile_logs) == []


@pytest.mark.asyncio
async def test_latches_are_per_operation(monkeypatch, profile_logs, clock):
    """One helper's failure must not silence an unrelated helper."""
    _fresh_latches(monkeypatch, clock)
    user_id = "user-two-operations"

    _use_redis(monkeypatch, _BrokenRedis())
    await user_profile.get_user_interactions(user_id)

    _use_redis(monkeypatch, _WorkingRedis())
    await user_profile.get_user_interactions(user_id)

    _use_redis(monkeypatch, _BrokenRedis())
    await user_profile.invalidate_user_profile(user_id)

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to get user interactions" in events[0][1]
    assert "get_user_interactions" in events[1][1]
    assert "Failed to invalidate user profile" in events[2][1]
    assert events[2][1] != events[0][1]


@pytest.mark.asyncio
async def test_pipeline_backed_helper_rearms_after_recovery(
    monkeypatch, profile_logs, clock
):
    """record_interaction's success falls off the end, so its latch must re-arm."""
    _fresh_latches(monkeypatch, clock)
    user_id = "user-pipeline"

    _use_redis(monkeypatch, _BrokenWriteRedis())
    await user_profile.record_interaction(user_id, 1)

    _use_redis(monkeypatch, _RecordableRedis())
    await user_profile.record_interaction(user_id, 1)

    # Two incidents minutes apart, not one flap: the window has to pass.
    clock.advance(REANNOUNCE_SECONDS + 1)
    _use_redis(monkeypatch, _BrokenWriteRedis())
    await user_profile.record_interaction(user_id, 1)

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to record user interaction" in events[0][1]
    assert "record_interaction" in events[1][1]
    assert "Failed to record user interaction" in events[2][1]


# --- transitions for the helpers that had no coverage of their own ---


@pytest.mark.asyncio
async def test_profile_vector_reader_rearms_after_recovery(
    monkeypatch, profile_logs, clock
):
    """get_user_profile_vector's own latch announces both outages.

    Only a stored vector counts as a success here; a missing key delegates to
    build_user_profile, which logs on its own latch.
    """
    _fresh_latches(monkeypatch, clock)
    user_id = "user-profile-vector"

    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_user_profile_vector(user_id) is None

    _use_redis(monkeypatch, _StoredVectorRedis())
    assert await user_profile.get_user_profile_vector(user_id) == [0.1, 0.2, 0.3]

    # Two incidents minutes apart, not one flap: the window has to pass.
    clock.advance(REANNOUNCE_SECONDS + 1)
    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_user_profile_vector(user_id) is None

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to get user profile vector" in events[0][1]
    assert "get_user_profile_vector" in events[1][1]
    assert "Failed to get user profile vector" in events[2][1]


@pytest.mark.asyncio
async def test_profile_builder_rearms_after_recovery(monkeypatch, profile_logs, clock):
    """build_user_profile's own latch announces both outages.

    A Redis outage cannot reach this latch -- it is absorbed by
    get_user_interactions, which has one of its own -- so what is driven up and
    down here is the article lookup the helper makes afterwards.
    """
    _fresh_latches(monkeypatch, clock)
    qdrant = _BrokenQdrant()
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    user_id = "user-profile-builder"

    _use_redis(monkeypatch, _InteractingRedis())
    qdrant.broken = True
    assert await user_profile.build_user_profile(user_id) is None

    qdrant.broken = False
    assert await user_profile.build_user_profile(user_id) == [0.1, 0.2, 0.3]

    # Two incidents minutes apart, not one flap: the window has to pass.
    clock.advance(REANNOUNCE_SECONDS + 1)
    qdrant.broken = True
    assert await user_profile.build_user_profile(user_id) is None

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to build user profile" in events[0][1]
    assert "build_user_profile" in events[1][1]
    assert "Failed to build user profile" in events[2][1]


@pytest.mark.asyncio
async def test_category_reader_rearms_after_recovery(monkeypatch, profile_logs, clock):
    """get_user_profile_categories' own latch announces both outages."""
    _fresh_latches(monkeypatch, clock)
    user_id = "user-categories"

    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_user_profile_categories(user_id) == []

    _use_redis(monkeypatch, _WorkingRedis())
    assert await user_profile.get_user_profile_categories(user_id) == []

    # Two incidents minutes apart, not one flap: the window has to pass.
    clock.advance(REANNOUNCE_SECONDS + 1)
    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_user_profile_categories(user_id) == []

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to get user categories" in events[0][1]
    assert "get_user_profile_categories" in events[1][1]
    assert "Failed to get user categories" in events[2][1]


@pytest.mark.asyncio
async def test_trending_reader_rearms_after_recovery(monkeypatch, profile_logs, clock):
    """get_trending_articles' own latch announces both outages."""
    _fresh_latches(monkeypatch, clock)

    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_trending_articles() == []

    # An empty scan is a clean window, which is what lets the recovery land here.
    _use_redis(monkeypatch, _WorkingRedis())
    assert await user_profile.get_trending_articles() == []

    # Two incidents minutes apart, not one flap: the window has to pass.
    clock.advance(REANNOUNCE_SECONDS + 1)
    _use_redis(monkeypatch, _BrokenRedis())
    assert await user_profile.get_trending_articles() == []

    events = _events(profile_logs)
    assert _levels(profile_logs) == [logging.WARNING] * 3
    assert "Failed to get trending articles" in events[0][1]
    assert "get_trending_articles" in events[1][1]
    assert "Failed to get trending articles" in events[2][1]
