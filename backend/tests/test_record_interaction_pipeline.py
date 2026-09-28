"""Behavioural tests for the `record_interaction` write transaction (#288).

The point of these tests is the property the source claims: a newly recorded
interaction evicts the user's cached profile vector and category aggregates *in
the same Redis transaction as the new signal*, so a reader can never observe a
derived value that predates the signal it was derived from.

To test that, the Redis double below is a stateful store rather than a mock:

* commands are applied to typed in-memory stores (`zsets`, `hashes`, `strings`),
  so every assertion is on data a real reader would go on to see;
* `execute()` is copy-on-write, so a failure part-way through a pipeline leaves
  the store exactly as it was. Note this is the double's model, not Redis's:
  a real `MULTI`/`EXEC` gives isolation, not rollback, so a command that fails
  at EXEC time still lets the rest of the batch commit.
* every buffered command is logged with the id of the pipeline that buffered it,
  because *which transaction a command rode in* is not observable from the
  resulting state: a delete issued in a second pipeline after `execute()` would
  leave the same final bytes as one issued in the same transaction.

Anything the double does not model raises `AssertionError` naming the command.
`record_interaction` swallows that, so the name reaches the captured log rather
than the test's own failure message -- the guarantee is only that the double
never silently no-ops.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app import user_profile
from app.main import state
from app.user_profile import (
    get_user_interactions,
    get_user_profile_categories,
    get_user_profile_vector,
    record_interaction,
)

USER = "user-1"
OTHER_USER = "user-2"
ARTICLE = 7
OTHER_ARTICLE = 9

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

    def zcard(self, key: str) -> None:
        # Buffered, never applied: the caller is the distinct-article cap check
        # (USER_MAX_DISTINCT_INTERACTIONS), which reads its answer off
        # `execute()` rather than off the committed stores.
        self._queue("zcard", key)

    def zscore(self, key: str, member: str) -> None:
        self._queue("zscore", key, str(member))

    def hincrby(self, key: str, field: str, amount: int) -> None:
        self._queue("hincrby", key, field, amount)

    def zincrby(self, key: str, amount: int, member: str) -> None:
        # Advances the trending index (#261) inside the write transaction.
        self._queue("zincrby", key, amount, str(member))

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
        results: list[Any] = []
        for method, args in batch:
            # `staged` is the only thing written, so a raise part-way through
            # leaves the committed stores exactly as they were.
            results.append(self._redis._apply(staged, method, args, count_executes=False))
        self._redis.commit(staged)
        return results


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
        # Expiries set with `SET ... EX` (the article-exists cache), kept apart
        # from `ttls` (expiries set with EXPIRE inside the write transaction).
        self.set_expiries: dict[str, int] = {}
        # (pipeline_id, method, args) for every buffered command, in order.
        self.commands: list[tuple[int, str, tuple[Any, ...]]] = []
        self.pipeline_calls = 0
        self.executes = 0

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

        zs, hs, st, tt = target["zsets"], target["hashes"], target["strings"], target["ttls"]
        if method == "zadd":
            key, mapping = args
            zs.setdefault(key, {}).update({str(m): float(s) for m, s in mapping.items()})
            return 1
        if method == "zincrby":
            key, amount, member = args
            zset = zs.setdefault(key, {})
            zset[str(member)] = zset.get(str(member), 0.0) + float(amount)
            return zset[str(member)]
        if method == "zcard":
            (key,) = args
            return len(zs.get(key, {}))
        if method == "zscore":
            key, member = args
            return zs.get(key, {}).get(member)
        if method == "hincrby":
            key, field, amount = args
            current = int(hs.setdefault(key, {}).get(field, "0"))
            hs[key][field] = str(current + int(amount))
            # Redis returns the POST-increment value, and #261 now depends on
            # it: record_interaction reads the first result to learn whether
            # these counters were just created, which decides whether the
            # trending index must be re-seeded. A constant here would make
            # every write look like a fresh counter.
            return current + int(amount)
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

    # ---- async commands (the shape the real client exposes) ------------------

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def hgetall(self, key: str) -> dict[str, str]:
        # Read by the trending-index re-seed (#261) after a pipeline reports
        # that the article's counters were just created.
        return dict(self.hashes.get(key, {}))

    async def exists(self, key: str) -> int:
        # Read by the trending-index bootstrap (#261).
        return int(key in self.strings or key in self.hashes or key in self.zsets)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """`SET ... EX`, as awaited by the article-exists negative cache.

        The expiry is kept in `set_expiries` rather than `ttls`, because
        `ttls` is what the EXPIRE-buffered write transaction is asserted
        against: folding a pre-transaction cache key in there would make that
        assertion cover a key the transaction never touched.
        """
        self.strings[key] = value
        if ex is not None:
            self.set_expiries[key] = int(ex)
        return True

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


class _FakeQdrant:
    """Answers the pre-write article-exists guard with a known article.

    `record_interaction` refuses an article the index does not hold, so this
    double confirms every id it is asked about. That guard is not what these
    tests pin -- it has its own coverage -- and an unmodelled index here would
    make `record_interaction` decline before the write pipeline is ever built.
    """

    def __init__(self, unknown_ids: set[int] | None = None) -> None:
        self.unknown_ids = unknown_ids or set()
        self.asked: list[list[int]] = []

    async def retrieve(self, *, ids: list[int], **kwargs: Any) -> list[Any]:
        self.asked.append(list(ids))
        return [SimpleNamespace(id=i) for i in ids if i not in self.unknown_ids]


@pytest.fixture
def qdrant(monkeypatch) -> _FakeQdrant:
    """An index that knows every article, installed on the app state."""
    fake = _FakeQdrant()
    monkeypatch.setitem(state, "qdrant", fake)
    return fake


@pytest.fixture
def redis(monkeypatch, qdrant) -> _FakeRedis:
    """Point the module at a fresh in-memory Redis; no network, no env deps.

    The distinct-article cap is pinned so its guard always runs the same way:
    an ambient `USER_MAX_DISTINCT_INTERACTIONS=0` would skip the check, and
    these tests would then cover a write pipeline that is not the real one.
    """
    fake = _FakeRedis()
    monkeypatch.setattr(user_profile, "_redis_client", lambda: fake)
    monkeypatch.setattr(user_profile.config, "USER_MAX_DISTINCT_INTERACTIONS", 500)
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

    # The writer also maintains the trending index (#261), so this asserts the
    # reader's key is there and correct rather than that it is the only zset.
    assert f"user:interactions:{USER}" in redis.zsets
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

    A click then a read on article 7 leave `click=1, read=1`. A third click on
    a different article must land in its own counter key, so a key that dropped
    the article id would merge the two articles' counts and fail here.
    """
    await record_interaction(USER, ARTICLE, interaction_type="click")
    await record_interaction(USER, ARTICLE, interaction_type="read")
    await record_interaction(USER, OTHER_ARTICLE, interaction_type="click")

    counters = redis.hashes[f"article:interactions:{ARTICLE}"]
    assert counters["click"] == "1"
    assert counters["read"] == "1"
    assert float(counters["last_timestamp"]) == pytest.approx(redis.zsets[f"user:interactions:{USER}"]["7"])
    # The other article's counter is separate, so nothing bled across.
    assert redis.hashes[f"article:interactions:{OTHER_ARTICLE}"]["click"] == "1"
    # Exactly two article counters exist, so no count bled across articles.
    assert {key for key in redis.hashes if key.startswith("article:interactions:")} == {
        f"article:interactions:{ARTICLE}",
        f"article:interactions:{OTHER_ARTICLE}",
    }


# ---- 3. TTLs on every written key -------------------------------------------


@pytest.mark.asyncio
async def test_every_written_key_receives_its_ttl(redis, monkeypatch):
    """Every written key gets a TTL, asserted exactly.

    The interaction set is pinned to the module's one-year raw-signal horizon
    (it exists so profile building has long-term history to read); the detail
    hash, the counter hash and the trending index (#261) share the configured
    TTL, pinned here so the assertion is independent of the ambient
    environment. The index and its ready marker deliberately carry the COUNTER
    TTL, so neither can outlive the counters it ranks or the seed state that
    backfills it. The 365 is written out rather than imported, so shortening
    the retention horizon has to update this test.
    """
    monkeypatch.setattr(user_profile.config, "USER_INTERACTION_TTL_DAYS", CONTROLLED_TTL_DAYS)

    await record_interaction(USER, ARTICLE)

    assert redis.ttls == {
        f"user:interactions:{USER}": INTERACTION_SET_TTL_DAYS * 86400,
        f"user:interaction_detail:{USER}:{ARTICLE}": CONTROLLED_TTL_DAYS * 86400,
        f"article:interactions:{ARTICLE}": CONTROLLED_TTL_DAYS * 86400,
        user_profile._TRENDING_INDEX_KEY: CONTROLLED_TTL_DAYS * 86400,
        user_profile._TRENDING_INDEX_READY_KEY: CONTROLLED_TTL_DAYS * 86400,
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
    """One write pipeline, one execute, and the delete shares its pipeline id.

    Transaction identity is not visible in the final state -- deleting in a
    second pipeline after `execute()` would leave identical bytes -- so it is
    read off the command log.

    Two other pipelines may appear. One is read-only and precedes the write: the
    distinct-article cap check asks ZCARD/ZSCORE. The other is the trending
    index re-seed (#261), which can only learn that the article's counters are
    brand new FROM the write's own result, so it is necessarily a post-commit
    repair. It may write, but only to the index. Splitting the write in two, or
    moving the delete into its own pipeline, still fails both halves.
    """
    await record_interaction(USER, ARTICLE)

    ids_by_method: dict[str, set[int]] = {}
    for pipeline_id, method, _args in redis.commands:
        ids_by_method.setdefault(method, set()).add(pipeline_id)

    # The article counter is buffered by exactly one pipeline, and that is the
    # write. The trending index advances in it too, so the index can never lag
    # a committed counter: both land in the same MULTI/EXEC or neither does.
    write_ids = ids_by_method["hincrby"]
    assert len(write_ids) == 1, (
        f"the whole write must be buffered by one pipeline; saw {write_ids} for hincrby "
        f"across methods {ids_by_method}"
    )
    write_id = write_ids.pop()
    assert ids_by_method["zincrby"] == {write_id}, (
        "the trending index must advance in the same transaction as the counters, "
        f"not in a pipeline of its own; seen pipeline ids per command: {ids_by_method}"
    )

    # Every other pipeline either only reads, or is the post-commit index
    # re-seed, which is allowed to write but only to the index.
    other_ids = {pid for pid, _method, _args in redis.commands} - {write_id}
    for pipeline_id in other_ids:
        buffered = [(m, a) for pid, m, a in redis.commands if pid == pipeline_id]
        methods = {m for m, _a in buffered}
        assert methods <= {"zcard", "zscore", "zadd", "expire"}, (
            f"pipeline {pipeline_id} carries writes {sorted(methods - {'zcard', 'zscore'})}"
        )
        for _m, args in buffered:
            if _m in {"zadd", "expire"}:
                assert args[0] == user_profile._TRENDING_INDEX_KEY, (
                    "only the trending index may be repaired after the write commits; "
                    f"pipeline {pipeline_id} touched {args[0]!r}"
                )
    assert redis.pipeline_calls == len(other_ids) + 1
    assert redis.executes == redis.pipeline_calls, "each pipeline must be executed exactly once"

    assert ids_by_method.get("delete") == {write_id}, (
        "the derived-key delete must be buffered by the same pipeline as the counters; "
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


