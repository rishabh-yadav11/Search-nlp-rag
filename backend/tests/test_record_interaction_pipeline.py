"""`record_interaction` must evict the cached profile in the same Redis
transaction as the new signal.

Counters must advance atomically (`zincrby`/`hincrby`, never a read-modify-write),
and transaction identity is not observable from the final state, so the double
logs the pipeline id that buffered each command.
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

INTERACTION_SET_TTL_DAYS = 365
#: Pinned so the asserted TTL seconds never depend on the ambient environment.
CONTROLLED_TTL_DAYS = 7


def _unmodelled(where: str, name: str) -> AssertionError:
    return AssertionError(f"{where} does not model Redis command {name!r}")


class _FakePipeline:
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
        self._queue("zcard", key)

    def zscore(self, key: str, member: str) -> None:
        self._queue("zscore", key, str(member))

    def hincrby(self, key: str, field: str, amount: int) -> None:
        self._queue("hincrby", key, field, amount)

    def zincrby(self, key: str, amount: int, member: str) -> None:
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
        if name.startswith("_"):
            raise AttributeError(name)
        raise _unmodelled("_FakePipeline", name)

    async def execute(self) -> list[Any]:
        self._redis.executes += 1
        batch = list(self.pending)
        self.pending.clear()
        staged = self._redis.snapshot()
        results: list[Any] = []
        for method, args in batch:
            # Only `staged` is written, so a mid-batch raise leaves the committed stores untouched.
            results.append(self._redis._apply(staged, method, args, count_executes=False))
        self._redis.commit(staged)
        return results


class _FakeRedis:
    """Stores are split per type so a `zadd` cannot satisfy a `get`, which is what
    makes a wrong key prefix fail as missing data instead of passing.
    """

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, float]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        # Expiries set by `SET ... EX`, kept apart from `ttls` (EXPIRE inside the write).
        self.set_expiries: dict[str, int] = {}
        self.commands: list[tuple[int, str, tuple[Any, ...]]] = []
        self.pipeline_calls = 0
        self.executes = 0


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
            # Redis returns the post-increment value; the source reads it to detect brand-new counters.
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


    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def exists(self, key: str) -> int:
        return int(key in self.strings or key in self.hashes or key in self.zsets)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
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
    vector_key = f"user:profile_vector:{user_id}"
    categories_key = f"user:categories:{user_id}"
    vector = [0.1, 0.2, 0.3]
    redis.seed_string(vector_key, json.dumps(vector))
    redis.seed_zset(categories_key, {"technology": 1.0, "merger": 0.5})
    return {"vector_key": vector_key, "categories_key": categories_key, "vector": vector}


class _FakeQdrant:
    """Every requested id is known: `record_interaction` refuses an article the
    index does not hold, so an unmodelled index would decline the write.
    """

    def __init__(self, unknown_ids: set[int] | None = None) -> None:
        self.unknown_ids = unknown_ids or set()
        self.asked: list[list[int]] = []

    async def retrieve(self, *, ids: list[int], **kwargs: Any) -> list[Any]:
        self.asked.append(list(ids))
        return [SimpleNamespace(id=i) for i in ids if i not in self.unknown_ids]


@pytest.fixture
def qdrant(monkeypatch) -> _FakeQdrant:
    fake = _FakeQdrant()
    monkeypatch.setitem(state, "qdrant", fake)
    return fake


@pytest.fixture
def redis(monkeypatch, qdrant) -> _FakeRedis:
    """Pins the distinct-article cap: an ambient 0 would skip the check, leaving
    these tests covering a different pipeline.
    """
    fake = _FakeRedis()
    monkeypatch.setattr(user_profile, "_redis_client", lambda: fake)
    monkeypatch.setattr(user_profile.config, "USER_MAX_DISTINCT_INTERACTIONS", 500)
    return fake


@pytest.mark.asyncio
async def test_recording_an_interaction_is_visible_to_the_interactions_reader(redis):
    await record_interaction(USER, 42)

    # The writer also maintains the trending index, so this pins the reader's key is present, not that it is alone.
    assert f"user:interactions:{USER}" in redis.zsets
    assert list(redis.zsets[f"user:interactions:{USER}"]) == ["42"], (
        "the sorted set member is the article id as a string"
    )

    recorded = await get_user_interactions(USER)
    assert [article_id for article_id, _ in recorded] == [42]
    assert recorded[0][1] == pytest.approx(redis.zsets[f"user:interactions:{USER}"]["42"])


@pytest.mark.asyncio
async def test_article_counters_are_keyed_by_article_and_accumulate_per_type(redis):
    await record_interaction(USER, ARTICLE, interaction_type="click")
    await record_interaction(USER, ARTICLE, interaction_type="read")
    await record_interaction(USER, OTHER_ARTICLE, interaction_type="click")

    counters = redis.hashes[f"article:interactions:{ARTICLE}"]
    assert counters["click"] == "1"
    assert counters["read"] == "1"
    assert float(counters["last_timestamp"]) == pytest.approx(redis.zsets[f"user:interactions:{USER}"]["7"])
    assert redis.hashes[f"article:interactions:{OTHER_ARTICLE}"]["click"] == "1"
    assert {key for key in redis.hashes if key.startswith("article:interactions:")} == {
        f"article:interactions:{ARTICLE}",
        f"article:interactions:{OTHER_ARTICLE}",
    }


@pytest.mark.asyncio
async def test_every_written_key_receives_its_ttl(redis, monkeypatch):
    """The 365-day interaction set is written out, not imported, so shortening the horizon must update this test."""
    monkeypatch.setattr(user_profile.config, "USER_INTERACTION_TTL_DAYS", CONTROLLED_TTL_DAYS)

    await record_interaction(USER, ARTICLE)

    assert redis.ttls == {
        f"user:interactions:{USER}": INTERACTION_SET_TTL_DAYS * 86400,
        f"user:interaction_detail:{USER}:{ARTICLE}": CONTROLLED_TTL_DAYS * 86400,
        f"article:interactions:{ARTICLE}": CONTROLLED_TTL_DAYS * 86400,
        user_profile._TRENDING_INDEX_KEY: CONTROLLED_TTL_DAYS * 86400,
        user_profile._TRENDING_INDEX_READY_KEY: CONTROLLED_TTL_DAYS * 86400,
    }


@pytest.mark.asyncio
async def test_recording_an_interaction_evicts_the_cached_derived_profile(redis, monkeypatch):
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
    seeded = _seed_derived_profile(redis, USER)

    await record_interaction(USER, ARTICLE)

    assert seeded["categories_key"] not in redis.zsets
    assert await get_user_profile_categories(USER) == []


@pytest.mark.asyncio
async def test_the_derived_key_deletes_ride_in_the_same_transaction_as_the_write(redis):
    """Transaction identity is not visible in the final state, so it is read off the pipeline id."""
    await record_interaction(USER, ARTICLE)

    ids_by_method: dict[str, set[int]] = {}
    for pipeline_id, method, _args in redis.commands:
        ids_by_method.setdefault(method, set()).add(pipeline_id)

    # The counter and the trending index share the one write pipeline, so the index can never lag a committed counter.
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

    # Tolerated extras: a read-only distinct-article cap check, and a trending-index
    # re-seed that can only learn the counters are new from the write's own result.
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


@pytest.mark.asyncio
async def test_eviction_is_scoped_to_the_recording_user(redis):
    other = _seed_derived_profile(redis, OTHER_USER)

    await record_interaction(USER, ARTICLE)

    assert redis.strings[other["vector_key"]] == json.dumps(other["vector"])
    assert redis.zsets[other["categories_key"]] == {"technology": 1.0, "merger": 0.5}


