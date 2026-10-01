"""Trending reads against an in-memory Redis stand-in (no fakeredis dependency) that records sequential HGETALL round trips and SCANs."""

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
                # Lets a test land a real write between the index read and that
                # batch's hydration.
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
    """Minimal async Redis stand-in tracking round trips."""

    def __init__(self):
        self.store: dict = {}
        self.zsets: dict = {}
        self.scan_calls = 0
        self.hgetall_round_trips: list[str] = []
        self.pipeline_hgetall_batches: list[list[str]] = []
        # Set by a test that needs to interleave a write with a read.
        self.during_hgetall_batch = None
        self.ttls: dict = {}

    # --- command execution, sync like redis-py's builders ---
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

    # --- awaited commands ---
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
        # Redis orders ties reverse-lexicographically, so sort (score, member)
        # descending: reversing an ascending sort would give reverse-lexicographic
        # ties but ASCENDING scores.
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        ordered = [m for m, _ in items]
        return ordered[start:stop + 1]


def _run(coro):
    return asyncio.run(coro)


async def _always_ok(*_args, **_kwargs):
    """Stand-in for the Qdrant article-existence guard (see the `fake` fixture)."""


@pytest.fixture
def fake(monkeypatch):
    """A bare in-memory Redis, with the two write guards stood down.

    ``record_interaction`` refuses an article missing from the Qdrant index and a
    user over the distinct-article cap, both of which need a collection this fake
    does not carry. A cap of 0 short-circuits the slot check before it issues
    ZCARD/ZSCORE, which this fake does not model.
    """
    redis = _FakeRedis()
    monkeypatch.setattr(user_profile, "_redis_client", lambda: redis)
    monkeypatch.setattr(user_profile, "_require_known_article", _always_ok)
    monkeypatch.setattr(user_profile.config, "USER_MAX_DISTINCT_INTERACTIONS", 0)
    return redis


def _seed_interactions(fake, interactions):
    """Record interactions and return {article_id: interaction_count}."""
    counts: dict[int, int] = {}
    for article_id, kind in interactions:
        _run(record_interaction("u1", article_id, kind))
        counts[article_id] = counts.get(article_id, 0) + 1
    return counts


def _oracle(counts, limit):
    """Expected trending output: rank by interaction count, take `limit`."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [{"article_id": aid, "score": float(n)} for aid, n in ranked]


def _legacy_full_scan_ranking(fake, limit):
    """The old ranking, transcribed: SCAN the keyspace, HGETALL every
    `article:interactions:*` key, sum its digit-valued counters, rank by total
    descending, cut to `limit`. Sorted key order makes the comparison exact."""
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
    """Take the one-time bootstrap scan, then clear the round-trip recorders."""
    _drop_window_cache(fake)
    _run(get_trending_articles(limit=limit))
    _drop_window_cache(fake)
    fake.scan_calls = 0
    fake.hgetall_round_trips.clear()
    fake.pipeline_hgetall_batches.clear()


# --- equivalence with the old full-scan ranking ---


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
    # The window cache key carries no limit, so drop it before asking for another.
    _drop_window_cache(fake)
    assert _run(get_trending_articles(limit=1)) == _oracle(counts, 1)


def test_trending_matches_the_pre_fix_full_scan_ranking(fake):
    """Differential check against the full-scan ranking this replaced."""
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
    """Scores come from the counters themselves, one field per interaction type."""
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
    """An indexed article with no counters left is skipped, not ranked."""
    _seed_interactions(fake, [(1, "click"), (2, "click"), (2, "click"), (3, "click")])
    _warm_up_index(fake, limit=5)
    del fake.store["article:interactions:1"]

    out = _run(get_trending_articles(limit=2))

    assert out == [{"article_id": 2, "score": 2.0}, {"article_id": 3, "score": 1.0}]


def test_trending_pages_past_a_fully_expired_rank_batch(fake):
    """An entire expired top batch must not starve the result set."""
    _seed_interactions(fake, [(i, "click") for i in range(1, 61)])
    _warm_up_index(fake, limit=10)
    for article_id in range(1, 51):
        del fake.store[f"article:interactions:{article_id}"]

    out = _run(get_trending_articles(limit=10))

    assert out == [{"article_id": aid, "score": 1.0} for aid in range(51, 61)]


# --- no keyspace walk on the read path ---


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


# --- per-key reads are batched, not one round trip per key ---


def test_candidate_hydration_is_batched_not_serial(fake):
    counts = _seed_interactions(fake, [(i, "click") for i in range(1, 121)])
    _warm_up_index(fake, limit=120)

    out = _run(get_trending_articles(limit=120))

    assert out == _oracle(counts, 120)
    # The serial path awaited HGETALL once per key.
    assert fake.hgetall_round_trips == []
    assert [len(b) for b in fake.pipeline_hgetall_batches if b] == [50, 50, 20]
    assert fake.scan_calls == 0


def test_read_cost_does_not_grow_with_index_size(fake):
    """Reads stay bounded by the rank batch, not by how many articles exist."""
    _seed_interactions(fake, [(i, "click") for i in range(1, 401)])
    _warm_up_index(fake, limit=10)

    _run(get_trending_articles(limit=10))

    hydrated = [b for b in fake.pipeline_hgetall_batches if b]
    assert len(hydrated) == 1
    assert len(hydrated[0]) == 50  # one rank batch, not 400 keys
    assert sum(1 for k in fake.store if k.startswith("article:interactions:")) == 400


# --- write path keeps the index in step ---


def test_repeat_interactions_advance_the_index_incrementally(fake):
    """The index must track repeats, not only the first hit of each counter."""
    _seed_interactions(
        fake, [(1, "click"), (1, "click"), (1, "click"), (2, "click"), (2, "click")]
    )

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 3.0, "2": 2.0}


def test_record_interaction_advances_the_trending_index(fake):
    _seed_interactions(fake, [(1, "click"), (1, "view"), (2, "click")])

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 2.0, "2": 1.0}


def test_recreated_counters_reseed_the_stale_index_score(fake):
    """After counters expire, the next interaction must not inherit the old score."""
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"5": 99.0}
    fake.store[user_profile._TRENDING_INDEX_READY_KEY] = "1"

    _run(record_interaction("u1", 5, "click"))

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"5": 1.0}


def test_recreated_counters_reseed_multi_type_article(fake):
    """A recreated hash holding other counters is re-seeded with the real total."""
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"5": 99.0}
    fake.store["article:interactions:5"] = {"view": "4"}

    _run(record_interaction("u1", 5, "click"))

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"5": 5.0}


def test_repeat_interactions_of_one_type_do_not_reseed(fake):
    """A counter that already exists must not trigger the re-seed read."""
    _seed_interactions(fake, [(1, "click")])
    reseeds_after_first = len(fake.hgetall_round_trips)

    _run(record_interaction("u1", 1, "click"))
    _run(record_interaction("u1", 1, "click"))

    assert reseeds_after_first == 1
    assert len(fake.hgetall_round_trips) == 1
    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 3.0}


def test_a_new_interaction_type_still_reports_the_true_total(fake):
    """The first interaction of a type re-seeds, and the total stays exact."""
    _seed_interactions(fake, [(1, "click"), (1, "click"), (2, "click")])
    _drop_window_cache(fake)

    _run(record_interaction("u2", 1, "view"))

    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == {"1": 3.0, "2": 1.0}
    _drop_window_cache(fake)
    assert _run(get_trending_articles(limit=5)) == _legacy_full_scan_ranking(fake, 5)


def test_write_path_never_marks_the_index_seeded(fake):
    """Interactions alone must not suppress the one-time bootstrap scan.

    A legacy install that records an interaction before its first trending read
    would otherwise be marked seeded without ever being scanned.
    """
    _seed_interactions(fake, [(1, "click"), (2, "click"), (2, "click"), (3, "click")])

    assert user_profile._TRENDING_INDEX_READY_KEY not in fake.store
    assert _run(get_trending_articles(limit=10)) == _legacy_full_scan_ranking(fake, 10)


# --- cache semantics preserved ---


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


# --- a live ingest must not be lost by a read already in flight --------------


def test_article_interacted_with_mid_read_still_appears(fake):
    """An ingest landing after the index read is not dropped by that read.

    The hook fires while the FIRST batch is hydrated, so ZREVRANGE has already
    returned without article 9; the loop must re-read the index for its next
    batch rather than sorting a snapshot taken before the write.
    """
    # Distinct scores put the new article at the BOTTOM of the index, the case a
    # rank cursor advances past safely; its click must come from the next batch.
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



# --- pre-index members: the backfill story ------------------------------------


def test_legacy_members_are_backfilled_lazily_on_the_first_read(fake):
    """Counters written by a deploy that predates the index are not lost.

    There is NO migration: one scan on the first trending read, guarded by a
    ready marker that keeps it from repeating, discovers every pre-existing
    `article:interactions:*` hash exactly as the old keyspace walk did.
    """
    # A pre-index install: counters only, no index, no ready marker.
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


# --- the order is a total order, not a function of Redis tie-breaking --------


def test_tied_scores_have_a_deterministic_order(fake):
    """Equal scores are broken by article id, so identical requests agree.

    Real Redis breaks a ZREVRANGE score tie in reverse lexicographic order, the
    OPPOSITE of the required order, so the final sort must impose a total order.
    """
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
    """The tie-break comes from the sort, not from the index's member order.

    Redis returns tied members reverse-lexicographically -- 9 before 10 -- while
    the required answer is 9 first, so a string comparison would fail here.
    """
    fake.store["article:interactions:10"] = {"click": "1"}
    fake.store["article:interactions:9"] = {"click": "1"}
    fake.zsets[user_profile._TRENDING_INDEX_KEY] = {"10": 1.0, "9": 1.0}
    fake.store[user_profile._TRENDING_INDEX_READY_KEY] = "1"

    # The index really does return them in the unhelpful order.
    assert _run(_zrevrange_now(fake)) == ["9", "10"]

    out = _run(get_trending_articles(limit=5))

    assert out == [
        {"article_id": 9, "score": 1.0},
        {"article_id": 10, "score": 1.0},
    ]


def _zrevrange_now(fake):
    return fake.zrevrange(user_profile._TRENDING_INDEX_KEY, 0, 49)


# --- index lifecycle ---------------------------------------------------------


def test_ready_marker_shares_the_index_ttl_so_they_cannot_outlive_each_other(fake):
    """The seed marker expires with the index it guards.

    A permanent marker beside a TTL'd index would suppress the backfill forever
    once the index expired, stranding later installs on an empty trending set.
    """
    _seed_interactions(fake, [(1, "click")])
    _run(get_trending_articles(limit=5))

    assert fake.ttls[user_profile._TRENDING_INDEX_KEY] == (
        user_profile.config.USER_INTERACTION_TTL_DAYS * 86400
    )
    assert fake.ttls[user_profile._TRENDING_INDEX_READY_KEY] == (
        user_profile.config.USER_INTERACTION_TTL_DAYS * 86400
    )


def test_index_is_not_rebuilt_per_call(fake):
    """A warm read costs no scan and no re-seed, so the optimisation holds."""
    _seed_interactions(fake, [(i, "click") for i in range(1, 21)])
    _warm_up_index(fake, limit=10)
    index_after_warmup = dict(fake.zsets[user_profile._TRENDING_INDEX_KEY])

    for _ in range(3):
        _drop_window_cache(fake)
        _run(get_trending_articles(limit=10))

    assert fake.scan_calls == 0
    assert fake.zsets[user_profile._TRENDING_INDEX_KEY] == index_after_warmup
