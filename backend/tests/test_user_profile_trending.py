"""No fakeredis dependency: the stand-in records sequential HGETALL round trips and SCANs."""

import asyncio

import pytest

from app import user_profile
from app.user_profile import get_trending_articles, record_interaction


class _Pipeline:
    def __init__(self, redis):
        self._redis = redis
        self._cmds = []

    def set(self, key, value, ex=None):
        self._cmds.append(("set", key, value))
        return self

    def zadd(self, key, mapping):
        self._cmds.append(("zadd", key, mapping))
        return self

    def zincrby(self, key, amount, member):
        self._cmds.append(("zincrby", key, amount, member))
        return self

    def hset(self, key, field=None, value=None, mapping=None):
        if mapping is not None:
            self._cmds.append(("hset", key, mapping))
        else:
            self._cmds.append(("hset", key, {field: value}))
        return self

    def hincrby(self, key, field, amount):
        self._cmds.append(("hincrby", key, field, amount))
        return self

    def hgetall(self, key):
        self._cmds.append(("hgetall", key))
        return self

    def expire(self, key, seconds):
        self._cmds.append(("expire", key, seconds))
        return self

    def delete(self, *keys):
        self._cmds.append(("delete", *keys))
        return self

    async def execute(self):
        results = []
        for cmd in self._cmds:
            if cmd[0] == "hgetall":
                self._redis.pipeline_hgetall_batches[-1].append(cmd[1])
                # Fires during batch hydration: lets a test write after ZREVRANGE, before the sort.
                if self._redis.during_hgetall_batch is not None:
                    hook, self._redis.during_hgetall_batch = (
                        self._redis.during_hgetall_batch,
                        None,
                    )
                    await hook()
            results.append(self._redis._apply(cmd))
        self._cmds = []
        return results


class _FakeRedis:
    def __init__(self):
        self.store: dict = {}
        self.zsets: dict = {}
        self.scan_calls = 0
        self.hgetall_round_trips: list[str] = []
        self.pipeline_hgetall_batches: list[list[str]] = []
        self.during_hgetall_batch = None
        self.ttls: dict = {}

    def _apply(self, cmd):
        kind = cmd[0]
        if kind == "set":
            self.store[cmd[1]] = cmd[2]
            return True
        if kind == "zadd":
            zset = self.zsets.setdefault(cmd[1], {})
            added = sum(1 for m in cmd[2] if m not in zset)
            zset.update({m: float(s) for m, s in cmd[2].items()})
            self.store[cmd[1]] = "zset"
            return added  # redis returns the number of NEW members added
        if kind == "zincrby":
            zset = self.zsets.setdefault(cmd[1], {})
            zset[cmd[3]] = zset.get(cmd[3], 0.0) + cmd[2]
            self.store[cmd[1]] = "zset"
            return zset[cmd[3]]
        if kind == "hset":
            self.store.setdefault(cmd[1], {}).update(cmd[2])
            return 1
        if kind == "hincrby":
            field = self.store.setdefault(cmd[1], {})
            field[cmd[2]] = str(int(field.get(cmd[2], 0)) + cmd[3])
            return int(field[cmd[2]])
        if kind == "hgetall":
            return dict(self.store.get(cmd[1], {}))
        if kind == "expire":
            self.ttls[cmd[1]] = cmd[2]
            return True
        if kind == "delete":
            for key in cmd[1:]:
                self.store.pop(key, None)
                self.zsets.pop(key, None)
            return len(cmd[1:])
        raise AssertionError(f"unexpected command {kind}")

    def pipeline(self):
        self.pipeline_hgetall_batches.append([])
        return _Pipeline(self)

    async def hgetall(self, key):
        self.hgetall_round_trips.append(key)
        return dict(self.store.get(key, {}))

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    async def exists(self, key):
        return 1 if key in self.store else 0

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
            self.zsets.pop(key, None)

    async def scan(self, cursor, match=None, count=None):
        self.scan_calls += 1
        prefix = match[:-1] if match and match.endswith("*") else None
        keys = sorted(k for k in self.store if prefix is None or k.startswith(prefix))
        size = count or 10
        page = keys[cursor:cursor + size]
        nxt = cursor + size
        return (0 if nxt >= len(keys) else nxt, page)

    async def zrevrange(self, key, start, stop, withscores=False):
        # Redis breaks score ties in REVERSE lexicographic member order.
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        ordered = [m for m, _ in items]
        return ordered[start:stop + 1]


def _run(coro):
    return asyncio.run(coro)


async def _always_ok(*_args, **_kwargs):
    """Stand-in for the Qdrant article-existence guard."""


@pytest.fixture
def fake(monkeypatch):
    """A cap of 0 short-circuits the cap check before ZCARD/ZSCORE, which this fake cannot model."""
    redis = _FakeRedis()
    monkeypatch.setattr(user_profile, "_redis_client", lambda: redis)
    monkeypatch.setattr(user_profile, "_require_known_article", _always_ok)
    monkeypatch.setattr(user_profile.config, "USER_MAX_DISTINCT_INTERACTIONS", 0)
    return redis


def _seed_interactions(fake, interactions):
    counts: dict[int, int] = {}
    for article_id, kind in interactions:
        _run(record_interaction("u1", article_id, kind))
        counts[article_id] = counts.get(article_id, 0) + 1
    return counts


def _oracle(counts, limit):
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [{"article_id": aid, "score": float(n)} for aid, n in ranked]


def _legacy_full_scan_ranking(fake, limit):
    """Ties fall out of SCAN order; this fake scans sorted, so equality comparison is exact."""
    article_scores: dict[str, float] = {}
    for key in sorted(fake.store):
        if not key.startswith("article:interactions:"):
            continue
        counts = fake.store[key]
        total = sum(int(v) for v in counts.values() if str(v).isdigit())
        if total > 0:
            article_scores[key.rsplit(":", 1)[-1]] = float(total)
    ranked = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [{"article_id": int(aid), "score": score} for aid, score in ranked]


def _drop_window_cache(fake):
    for key in [k for k in fake.store if k.startswith("trending:window:")]:
        del fake.store[key]


def _warm_up_index(fake, limit=10):
    _drop_window_cache(fake)
    _run(get_trending_articles(limit=limit))
    _drop_window_cache(fake)
    fake.scan_calls = 0
    fake.hgetall_round_trips.clear()
    fake.pipeline_hgetall_batches.clear()


def test_trending_matches_interaction_counts_for_known_set(fake):
    counts = _seed_interactions(
        fake,
        [(1, "click"), (2, "click"), (2, "click"), (2, "view"), (3, "click"),
         (4, "view"), (4, "read"), (4, "click"), (5, "click"), (6, "click")],
    )

    out = _run(get_trending_articles(limit=4))

    assert out == _oracle(counts, 4)
    assert out == [
        {"article_id": 2, "score": 3.0},
        {"article_id": 4, "score": 3.0},
        {"article_id": 1, "score": 1.0},
        {"article_id": 3, "score": 1.0},
    ]


def test_trending_limit_is_honoured(fake):
    counts = _seed_interactions(fake, [(i, "click") for i in range(1, 21)])

    assert _run(get_trending_articles(limit=3)) == _oracle(counts, 3)
    # The window cache key carries no limit; drop it before asking for a different one.
    _drop_window_cache(fake)
    assert _run(get_trending_articles(limit=1)) == _oracle(counts, 1)


def test_trending_matches_the_pre_fix_full_scan_ranking(fake):
    _seed_interactions(
        fake,
        [(1, "click"), (2, "click"), (2, "click"), (2, "view"), (3, "click"),
         (4, "view"), (4, "read"), (4, "click"), (5, "click"), (6, "click")],
    )

    for limit in (1, 3, 4, 6, 10, 20):
        _drop_window_cache(fake)
        out = _run(get_trending_articles(limit=limit))
        assert out == _legacy_full_scan_ranking(fake, limit), f"limit={limit}"


def test_trending_sums_every_interaction_type(fake):
    for article_id, clicks, views in [(7, 5, 2), (8, 3, 1), (9, 1, 0)]:
        fake.store[f"article:interactions:{article_id}"] = {
            "click": str(clicks), "view": str(views), "last_timestamp": "1700000000.5",
        }
    fake.store[user_profile._TRENDING_INDEX_READY_KEY] = "1"
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"7": 7.0, "8": 4.0, "9": 1.0}

    out = _run(get_trending_articles(limit=5))

    assert out == [
        {"article_id": 7, "score": 7.0},
        {"article_id": 8, "score": 4.0},
        {"article_id": 9, "score": 1.0},
    ]


def test_trending_skips_articles_whose_counters_expired(fake):
    _seed_interactions(fake, [(1, "click"), (2, "click"), (2, "click"), (3, "click")])
    _warm_up_index(fake, limit=5)
    del fake.store["article:interactions:1"]

    out = _run(get_trending_articles(limit=2))

    assert out == [{"article_id": 2, "score": 2.0}, {"article_id": 3, "score": 1.0}]


def test_trending_pages_past_a_fully_expired_rank_batch(fake):
    _seed_interactions(fake, [(i, "click") for i in range(1, 61)])
    _warm_up_index(fake, limit=10)
    for article_id in range(1, 51):
        del fake.store[f"article:interactions:{article_id}"]

    out = _run(get_trending_articles(limit=10))

    assert out == [{"article_id": aid, "score": 1.0} for aid in range(51, 61)]




def test_warm_trending_read_never_scans_the_keyspace(fake):
    counts = _seed_interactions(fake, [(1, "click"), (2, "click"), (2, "click")])
    _warm_up_index(fake, limit=5)

    out = _run(get_trending_articles(limit=5))

    assert out == _oracle(counts, 5)
    assert fake.scan_calls == 0


def test_bootstrap_runs_exactly_once_for_the_life_of_the_index(fake):
    _seed_interactions(fake, [(i, "click") for i in range(1, 31)])

    _run(get_trending_articles(limit=5))
    scans_after_first = fake.scan_calls
    for _ in range(3):
        _drop_window_cache(fake)
        _run(get_trending_articles(limit=5))

    assert scans_after_first >= 1
    assert fake.scan_calls == scans_after_first


def test_legacy_data_is_bootstrap_scanned_at_most_once(fake):
    for article_id in (1, 2, 3):
        fake.store[f"article:interactions:{article_id}"] = {"click": str(article_id)}

    first = _run(get_trending_articles(limit=3))
    scans_after_first = fake.scan_calls
    _drop_window_cache(fake)
    second = _run(get_trending_articles(limit=3))

    assert scans_after_first >= 1
    assert fake.scan_calls == scans_after_first
    assert first == second == [
        {"article_id": 3, "score": 3.0},
        {"article_id": 2, "score": 2.0},
        {"article_id": 1, "score": 1.0},
    ]


def test_empty_install_bootstraps_without_rescanning(fake):
    out = _run(get_trending_articles(limit=5))

    assert out == []
    assert fake.scan_calls >= 1
    del fake.store[user_profile._TRENDING_INDEX_READY_KEY]
    _run(get_trending_articles(limit=5))
    assert user_profile._TRENDING_INDEX_READY_KEY in fake.store




def test_candidate_hydration_is_batched_not_serial(fake):
    counts = _seed_interactions(fake, [(i, "click") for i in range(1, 121)])
    _warm_up_index(fake, limit=120)

    out = _run(get_trending_articles(limit=120))

    assert out == _oracle(counts, 120)
    assert fake.hgetall_round_trips == []
    assert [len(b) for b in fake.pipeline_hgetall_batches if b] == [50, 50, 20]
    assert fake.scan_calls == 0


def test_read_cost_does_not_grow_with_index_size(fake):
    _seed_interactions(fake, [(i, "click") for i in range(1, 401)])
    _warm_up_index(fake, limit=10)

    _run(get_trending_articles(limit=10))

    hydrated = [b for b in fake.pipeline_hgetall_batches if b]
    assert len(hydrated) == 1
    assert len(hydrated[0]) == 50
    assert sum(1 for k in fake.store if k.startswith("article:interactions:")) == 400




def test_repeat_interactions_advance_the_index_incrementally(fake):
    _seed_interactions(
        fake, [(1, "click"), (1, "click"), (1, "click"), (2, "click"), (2, "click")]
    )

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 3.0, "2": 2.0}


def test_record_interaction_advances_the_trending_index(fake):
    _seed_interactions(fake, [(1, "click"), (1, "view"), (2, "click")])

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 2.0, "2": 1.0}


def test_recreated_counters_reseed_the_stale_index_score(fake):
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"5": 99.0}
    fake.store[user_profile._TRENDING_INDEX_READY_KEY] = "1"

    _run(record_interaction("u1", 5, "click"))

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"5": 1.0}


def test_recreated_counters_reseed_multi_type_article(fake):
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"5": 99.0}
    fake.store["article:interactions:5"] = {"view": "4"}

    _run(record_interaction("u1", 5, "click"))

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"5": 5.0}


def test_repeat_interactions_of_one_type_do_not_reseed(fake):
    _seed_interactions(fake, [(1, "click")])
    reseeds_after_first = len(fake.hgetall_round_trips)

    _run(record_interaction("u1", 1, "click"))
    _run(record_interaction("u1", 1, "click"))

    assert reseeds_after_first == 1
    assert len(fake.hgetall_round_trips) == 1
    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 3.0}


def test_a_new_interaction_type_still_reports_the_true_total(fake):
    _seed_interactions(fake, [(1, "click"), (1, "click"), (2, "click")])
    _drop_window_cache(fake)

    _run(record_interaction("u2", 1, "view"))

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 3.0, "2": 1.0}
    _drop_window_cache(fake)
    assert _run(get_trending_articles(limit=5)) == _legacy_full_scan_ranking(fake, 5)


def test_write_path_never_marks_the_index_seeded(fake):
    """An interaction before the first trending read would set the ready marker without scanning."""
    _seed_interactions(fake, [(1, "click"), (2, "click"), (2, "click"), (3, "click")])

    assert user_profile._TRENDING_INDEX_READY_KEY not in fake.store
    assert _run(get_trending_articles(limit=10)) == _legacy_full_scan_ranking(fake, 10)




def test_window_cache_short_circuits_the_whole_read(fake):
    counts = _seed_interactions(fake, [(1, "click"), (2, "click"), (2, "click")])
    first = _run(get_trending_articles(limit=5))
    scans_after_first = fake.scan_calls
    fake.hgetall_round_trips.clear()

    window_keys = [k for k in fake.store if k.startswith("trending:window:")]
    assert len(window_keys) == 1
    second = _run(get_trending_articles(limit=5))

    assert second == first == _oracle(counts, 5)
    assert fake.hgetall_round_trips == []
    assert fake.scan_calls == scans_after_first




def test_article_interacted_with_mid_read_still_appears(fake):
    """Each batch must re-read the index, not sort a snapshot taken before the write."""
    # Distinct scores put the new article at the BOTTOM, the case a rank cursor advances past.
    _seed_interactions(fake, [(1, "click"), (1, "click"), (1, "click"),
                              (2, "click"), (2, "click")])
    _warm_up_index(fake, limit=5)
    _drop_window_cache(fake)

    async def ingest_mid_read():
        await record_interaction("u1", 9, "click")

    fake.during_hgetall_batch = ingest_mid_read
    out = _run(get_trending_articles(limit=5))
    fake.during_hgetall_batch = None

    assert out == [
        {"article_id": 1, "score": 3.0},
        {"article_id": 2, "score": 2.0},
        {"article_id": 9, "score": 1.0},
    ]




def test_legacy_members_are_backfilled_lazily_on_the_first_read(fake):
    for article_id in (1, 2, 3):
        fake.store[f"article:interactions:{article_id}"] = {"click": str(article_id)}
    assert user_profile._TRENDING_INDEX_KEY not in fake.zsets
    assert user_profile._TRENDING_INDEX_READY_KEY not in fake.store

    out = _run(get_trending_articles(limit=10))

    assert out == [
        {"article_id": 3, "score": 3.0},
        {"article_id": 2, "score": 2.0},
        {"article_id": 1, "score": 1.0},
    ]
    _run(record_interaction("u1", 4, "click"))
    _drop_window_cache(fake)
    assert {"article_id": 4, "score": 1.0} in _run(get_trending_articles(limit=10))




def test_tied_scores_have_a_deterministic_order(fake):
    for article_id in (3, 1, 4, 2):
        fake.store[f"article:interactions:{article_id}"] = {"click": "2"}
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"3": 2.0, "1": 2.0, "4": 2.0, "2": 2.0}
    fake.store[user_profile._TRENDING_INDEX_READY_KEY] = "1"

    runs = []
    for _ in range(5):
        _drop_window_cache(fake)
        runs.append(_run(get_trending_articles(limit=4)))

    assert runs[0] == [
        {"article_id": 1, "score": 2.0},
        {"article_id": 2, "score": 2.0},
        {"article_id": 3, "score": 2.0},
        {"article_id": 4, "score": 2.0},
    ]
    assert all(run == runs[0] for run in runs)


def test_tie_break_survives_a_reverse_lexicographic_index_order(fake):
    fake.store["article:interactions:10"] = {"click": "1"}
    fake.store["article:interactions:9"] = {"click": "1"}
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"10": 1.0, "9": 1.0}
    fake.store[user_profile._TRENDING_INDEX_READY_KEY] = "1"

    # The index hands them back in that order, so this is not vacuously sorted.
    assert _run(_zrevrange_now(fake)) == ["9", "10"]

    out = _run(get_trending_articles(limit=5))

    assert out == [
        {"article_id": 9, "score": 1.0},
        {"article_id": 10, "score": 1.0},
    ]


def _zrevrange_now(fake):
    return fake.zrevrange(user_profile._TRENDING_INDEX_KEY, 0, 49)




def test_ready_marker_shares_the_index_ttl_so_they_cannot_outlive_each_other(fake):
    """A permanent marker beside a TTL'd index would suppress the backfill forever."""
    _seed_interactions(fake, [(1, "click")])
    _run(get_trending_articles(limit=5))

    assert fake.ttls[user_profile._TRENDING_INDEX_KEY] == (
        user_profile.config.USER_INTERACTION_TTL_DAYS * 86400
    )
    assert fake.ttls[user_profile._TRENDING_INDEX_READY_KEY] == (
        user_profile.config.USER_INTERACTION_TTL_DAYS * 86400
    )


def test_index_is_not_rebuilt_per_call(fake):
    _seed_interactions(fake, [(i, "click") for i in range(1, 21)])
    _warm_up_index(fake, limit=10)
    index_after_warmup = dict(fake.zsets[user_profile._TRENDING_INDEX_KEY])

    for _ in range(3):
        _drop_window_cache(fake)
        _run(get_trending_articles(limit=10))

    assert fake.scan_calls == 0
    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == index_after_warmup
