"""Behavioural tests for the `record_interaction` write transaction (#288).

The point of these tests is the property the source claims: a newly recorded
interaction evicts the user's cached profile vector and category aggregates *in
the same Redis transaction as the new signal*, so a reader can never observe a
derived value that predates the signal it was derived from.

To test that, the Redis double below is a stateful store rather than a mock:

* commands are applied to typed in-memory stores (`zsets`, `hashes`, `strings`),
  so every assertion is on data a real reader would go on to see;
* `execute()` is copy-on-write, so a failure part-way through a pipeline leaves
  the store exactly as it was -- that is what makes "all or nothing" a
  checkable property rather than a comment;
* every buffered command is logged with the id of the pipeline that buffered it,
  because *which transaction a command rode in* is not observable from the
  resulting state: a delete issued in a second pipeline after `execute()` would
  leave the same final bytes as one issued in the same transaction.

Anything the double does not model raises `AssertionError` naming the command,
so a command added to the write path fails loudly instead of silently no-op'ing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app import user_profile
from app.user_profile import (
    get_user_interactions,
    get_user_profile_categories,
    get_user_profile_vector,
    record_interaction,
)

USER = "user-1"
OTHER_USER = "user-2"
ARTICLE = 7

#: Module constant: raw interactions are kept a full year for profile building.
INTERACTION_SET_TTL_DAYS = 365
#: Value substituted for the ambient `config.USER_INTERACTION_TTL_DAYS` in the
#: TTL test, so the asserted seconds never depend on the environment.
CONTROLLED_TTL_DAYS = 7


def _unmodelled(where: str, name: str) -> AssertionError:
    return AssertionError(f"{where} does not model Redis command {name!r}")


class _FakePipeline:
    """Buffers commands; touches no state until `execute()` succeeds.

    `hset` is called by the source both as `hset(key, mapping={...})` and as
    `hset(key, field, value)`, so both spellings are normalised to a single
    `(key, field, value)` triple in the pending list and in the command log.
    """

    def __init__(self, redis: _FakeRedis, pipeline_id: int) -> None:
        self._redis = redis
        self.pipeline_id = pipeline_id
        self.pending: list[tuple[str, tuple[Any, ...]]] = []

    def _queue(self, method: str, *args: Any) -> None:
        self._redis.commands.append((self.pipeline_id, method, args))
        self.pending.append((method, args))

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self._queue("zadd", key, dict(mapping))

    def hincrby(self, key: str, field: str, amount: int) -> None:
        self._queue("hincrby", key, field, amount)

    def hset(self, key: str, field: Any = None, value: Any = None, *, mapping: Any = None) -> None:
        if mapping is not None:
            pairs = list(mapping.items())
        else:
            pairs = [(field, value)]
        for name, val in pairs:
            self._queue("hset", key, name, str(val))

    def expire(self, key: str, seconds: int) -> None:
        self._queue("expire", key, seconds)

    def delete(self, *keys: str) -> None:
        self._queue("delete", keys)

    def __getattr__(self, name: str) -> Any:
        # `record_interaction` swallows every exception, so an unmodelled
        # command surfaces only through the captured log record. Raising a
        # named error puts the command in that message, so the tests below fail
        # on missing state *and* say which command the double failed to model.
        if name.startswith("_"):
            raise AttributeError(name)
        raise _unmodelled("_FakePipeline", name)

    async def execute(self) -> list[Any]:
        """Apply the buffered commands atomically, or not at all."""
        self._redis.executes += 1
        batch = list(self.pending)
        self.pending.clear()
        staged = self._redis.snapshot()
        for method, args in batch:
            # `staged` is the only thing written, so a raise part-way through
            # leaves the committed stores exactly as they were.
            self._redis._apply(staged, method, args, count_executes=False)
        self._redis.commit(staged)
        return [None] * len(batch)


class _FakeRedis:
    """Minimal stateful Redis covering the types the profile code actually uses.

    Stores are kept per type so a `zadd` cannot silently satisfy a `get` and
    vice versa, which is what makes a wrong key prefix show up as missing data
    rather than as a passing assertion.
    """

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, float]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        # (pipeline_id, method, args) for every buffered command, in order.
        self.commands: list[tuple[int, str, tuple[Any, ...]]] = []
        self.pipeline_calls = 0
        self.executes = 0
        #: Names of commands that raise a RuntimeError when *applied*. Used to
        #: prove a mid-transaction failure leaves the committed store untouched.
        self.fail_on: tuple[str, ...] = ()

    # ---- transaction plumbing -------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            "zsets": {k: dict(v) for k, v in self.zsets.items()},
            "hashes": {k: dict(v) for k, v in self.hashes.items()},
            "strings": dict(self.strings),
            "ttls": dict(self.ttls),
        }

    def commit(self, staged: dict[str, dict[str, Any]]) -> None:
        self.zsets = staged["zsets"]
        self.hashes = staged["hashes"]
        self.strings = staged["strings"]
        self.ttls = staged["ttls"]

    def pipeline(self) -> _FakePipeline:
        """Sync, as in redis-py: the returned object only buffers commands."""
        self.pipeline_calls += 1
        return _FakePipeline(self, self.pipeline_calls)

    def _apply(
        self,
        target: dict[str, dict[str, Any]],
        method: str,
        args: tuple[Any, ...],
        *,
        count_executes: bool = True,
    ) -> Any:
        if count_executes:
            self.executes += 1
        if method in self.fail_on:
            raise RuntimeError(f"injected Redis failure applying {method!r}")

        zs, hs, st, tt = target["zsets"], target["hashes"], target["strings"], target["ttls"]
        if method == "zadd":
            key, mapping = args
            zs.setdefault(key, {}).update({str(m): float(s) for m, s in mapping.items()})
            return 1
        if method == "hincrby":
            key, field, amount = args
            current = int(hs.setdefault(key, {}).get(field, "0"))
            hs[key][field] = str(current + int(amount))
            return 1
        if method == "hset":
            key, field, value = args
            hs.setdefault(key, {})[str(field)] = str(value)
            return 1
        if method == "expire":
            key, seconds = args
            tt[key] = int(seconds)
            return True
        if method == "delete":
            (keys,) = args
            removed = 0
            for key in keys:
                for store in (zs, hs, st):
                    if key in store:
                        del store[key]
                        removed += 1
                tt.pop(key, None)
            return removed
        raise _unmodelled("_FakeRedis._apply", method)

    # ---- async readers (the shape the real client exposes) -------------------

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def zrevrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False,
    ) -> list[Any]:
        members = sorted(self.zsets.get(key, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        if not members:
            return []
        first = max(0, min(start, len(members) - 1))
        last = len(members) - 1 if stop == -1 else min(stop, len(members) - 1)
        window = members[first : last + 1] if first <= last else []
        return window if withscores else [m for m, _ in window]

    async def delete(self, *keys: str) -> int:
        live = {"zsets": self.zsets, "hashes": self.hashes, "strings": self.strings, "ttls": self.ttls}
        return self._apply(live, "delete", (keys,), count_executes=False)

    # ---- sync setup helpers ---------------------------------------------------

    def set(self, key: str, value: str) -> None:
        self.strings[key] = value

    def seed_string(self, key: str, value: str) -> None:
        self.strings[key] = value

    def seed_zset(self, key: str, mapping: dict[str, float]) -> None:
        self.zsets[key] = {str(m): float(s) for m, s in mapping.items()}

    def seed_hash(self, key: str, mapping: dict[str, Any]) -> None:
        self.hashes[key] = {str(f): str(v) for f, v in mapping.items()}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise _unmodelled("_FakeRedis", name)


def _seed_derived_profile(redis: _FakeRedis, user_id: str) -> dict[str, Any]:
    """Populate the two derived keys a prior profile build would have cached."""
    vector_key = f"user:profile_vector:{user_id}"
    categories_key = f"user:categories:{user_id}"
    vector = [0.1, 0.2, 0.3]
    redis.seed_string(vector_key, json.dumps(vector))
    redis.seed_zset(categories_key, {"technology": 1.0, "merger": 0.5})
    return {"vector_key": vector_key, "categories_key": categories_key, "vector": vector}


@pytest.fixture
def redis(monkeypatch) -> _FakeRedis:
    """Point the module at a fresh in-memory Redis; no network, no env deps."""
    fake = _FakeRedis()
    monkeypatch.setattr(user_profile, "_redis_client", lambda: fake)
    return fake


# ---- 1. the signal lands where the readers look for it -----------------------


@pytest.mark.asyncio
async def test_recording_an_interaction_is_visible_to_the_interactions_reader(redis):
    """`get_user_interactions` returns the new signal as `(42, timestamp)`.

    The reader builds its own key from the same format string, so a changed
    prefix on the writer would leave the reader looking at a key that was never
    written and this assertion would fail.
    """
    await record_interaction(USER, 42)

    assert list(redis.zsets) == [f"user:interactions:{USER}"]
    assert list(redis.zsets[f"user:interactions:{USER}"]) == ["42"], (
        "the sorted set member is the article id as a string"
    )

    recorded = await get_user_interactions(USER)
    assert [article_id for article_id, _ in recorded] == [42]
    assert recorded[0][1] == pytest.approx(redis.zsets[f"user:interactions:{USER}"]["42"])


# ---- 2. article-level counters ----------------------------------------------


@pytest.mark.asyncio
async def test_article_counters_are_keyed_by_article_and_accumulate_per_type(redis):
    """Counts live at `article:interactions:{id}` and add up per type.

    A click then a read on article 7 leave `click=1, read=1`; article 9 is
    untouched, so a counter key that ignored the article id would fail.
    """
    await record_interaction(USER, ARTICLE, interaction_type="click")
    await record_interaction(USER, ARTICLE, interaction_type="read")

    counters = redis.hashes[f"article:interactions:{ARTICLE}"]
    assert counters["click"] == "1"
    assert counters["read"] == "1"
    assert float(counters["last_timestamp"]) == pytest.approx(redis.zsets[f"user:interactions:{USER}"]["7"])
    assert "article:interactions:9" not in redis.hashes


# ---- 3. TTLs on every written key -------------------------------------------


@pytest.mark.asyncio
async def test_every_written_key_receives_its_ttl(redis, monkeypatch):
    """All three written keys get a TTL, asserted exactly.

    The interaction set is pinned to the module's one-year raw-signal horizon
    (dwell-time analysis reads it); the detail and counter hashes use the
    configured TTL, pinned here so the assertion is independent of the ambient
    environment.
    """
    monkeypatch.setattr(user_profile.config, "USER_INTERACTION_TTL_DAYS", CONTROLLED_TTL_DAYS)

    await record_interaction(USER, ARTICLE)

    assert redis.ttls == {
        f"user:interactions:{USER}": INTERACTION_SET_TTL_DAYS * 86400,
        f"user:interaction_detail:{USER}:{ARTICLE}": CONTROLLED_TTL_DAYS * 86400,
        f"article:interactions:{ARTICLE}": CONTROLLED_TTL_DAYS * 86400,
    }


# ---- 4. the invariant: derived data is evicted, and cannot be read back -------


@pytest.mark.asyncio
async def test_recording_an_interaction_evicts_the_cached_derived_profile(redis, monkeypatch):
    """Both derived keys are gone, and a read rebuilds rather than serving stale.

    The stale vector is seeded, then a distinct replacement is returned by
    `build_user_profile`; `get_user_profile_vector` must hand back the
    replacement, which it can only do by missing the cache.
    """
    seeded = _seed_derived_profile(redis, USER)
    new_vector = [9.0, 9.0]

    async def fake_build(user_id: str) -> list[float]:
        return list(new_vector)

    monkeypatch.setattr(user_profile, "build_user_profile", fake_build)

    await record_interaction(USER, ARTICLE)

    assert seeded["vector_key"] not in redis.strings
    assert seeded["categories_key"] not in redis.zsets
    assert await get_user_profile_vector(USER) == new_vector
    assert await get_user_profile_vector(USER) != seeded["vector"]


@pytest.mark.asyncio
async def test_eviction_of_categories_is_observable_through_the_categories_reader(redis):
    """The category aggregate is cleared too, not just the vector."""
    seeded = _seed_derived_profile(redis, USER)

    await record_interaction(USER, ARTICLE)

    assert seeded["categories_key"] not in redis.zsets
    assert await get_user_profile_categories(USER) == []


# ---- 5. same transaction ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_derived_key_deletes_ride_in_the_same_transaction_as_the_write(redis):
    """One pipeline, one execute, and the delete shares their pipeline id.

    Transaction identity is not visible in the final state -- deleting in a
    second pipeline after `execute()` would leave identical bytes -- so it is
    read off the command log.
    """
    await record_interaction(USER, ARTICLE)

    assert redis.pipeline_calls == 1, "the whole write must be one pipeline"
    assert redis.executes == 1, "the pipeline must be executed exactly once"

    ids_by_method: dict[str, set[int]] = {}
    for pipeline_id, method, _args in redis.commands:
        ids_by_method.setdefault(method, set()).add(pipeline_id)

    assert ids_by_method.get("delete") == ids_by_method.get("zadd") == {1}, (
        "the derived-key delete must be buffered by the same pipeline as the zadd; "
        f"seen pipeline ids per command: {ids_by_method}"
    )

    delete_args = next(args for _pid, method, args in redis.commands if method == "delete")
    assert delete_args[0] == (f"user:profile_vector:{USER}", f"user:categories:{USER}")


# ---- 6. scoping ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_eviction_is_scoped_to_the_recording_user(redis):
    """Another user's derived keys survive an interaction by this user."""
    other = _seed_derived_profile(redis, OTHER_USER)

    await record_interaction(USER, ARTICLE)

    assert redis.strings[other["vector_key"]] == json.dumps(other["vector"])
    assert redis.zsets[other["categories_key"]] == {"technology": 1.0, "merger": 0.5}


# ---- 7. all or nothing --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_transaction_that_fails_commits_neither_the_signal_nor_the_eviction(redis):
    """A mid-transaction failure leaves the store exactly as it was.

    `record_interaction` swallows the error, so the only evidence is the state
    afterwards: the new signal is absent *and* the stale derived keys are still
    present. That is the "same transaction" claim under failure.
    """
    seeded = _seed_derived_profile(redis, USER)
    before = redis.snapshot()
    redis.fail_on = ("hincrby",)

    await record_interaction(USER, ARTICLE)  # the error is swallowed by design

    assert redis.executes >= 1, "the pipeline was attempted"
    assert redis.snapshot() == before
    assert await get_user_interactions(USER) == []
    assert redis.strings[seeded["vector_key"]] == json.dumps(seeded["vector"])
    assert seeded["categories_key"] in redis.zsets
