"""Every rejection here is paired with evidence that no Redis state was written, never a bare status code."""

import asyncio
import fnmatch
import random
import string
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import auth_cookie
from fastapi.testclient import TestClient
from qdrant_client.models import ScoredPoint

from app import auth as auth_module
from app import main, recommender, user_profile
from app.auth import AuthStore
from app.config import config
from app.user_profile import InteractionResult, InteractionType

# Anything outside KNOWN_IDS must be declined before a single key is written.
KNOWN_IDS = (101, 202, 303)
UNKNOWN_ID = 987654321

_LEGAL_KINDS = tuple(kind.value for kind in InteractionType)
_BOOKKEEPING_FIELDS = frozenset({"last_timestamp"})

# Sized as a flood loop: each unchecked distinct kind mints one permanent hash field.
_LONG_JUNK_KIND = "".join(
    random.Random(271).choice(string.ascii_letters + string.digits) for _ in range(8192)
)
_JUNK_KINDS = (
    "totally-made-up-kind",
    _LONG_JUNK_KIND,
) + tuple(f"junk-{i}-not-a-kind" for i in range(60))


class _FakeProfileRedis:
    """An unlisted command raises, so an unimplemented one fails the test instead of faking a passing guard."""

    def __init__(self):
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, int] = {}

    def key_dump(self) -> set[str]:
        """Every key that exists right now, across all three value types."""
        return set(self.strings) | set(self.hashes) | set(self.zsets)

    async def get(self, key):
        return self.strings.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return None
        self.strings[key] = str(value)
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def exists(self, key):
        # The trending bootstrap's one-time backfill check.
        return int(key in self.key_dump())

    async def delete(self, *keys):
        removed = 0
        for key in keys:
            removed += int(self.strings.pop(key, None) is not None)
            removed += int(self.hashes.pop(key, None) is not None)
            removed += int(self.zsets.pop(key, None) is not None)
        return removed

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    async def scan(self, cursor, match=None, count=None):
        keys = sorted(k for k in self.key_dump() if match is None or fnmatch.fnmatch(k, match))
        return 0, keys

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, field=None, value=None, mapping=None):
        target = self.hashes.setdefault(key, {})
        if mapping:
            target.update({str(k): str(v) for k, v in mapping.items()})
        if field is not None:
            target[str(field)] = str(value)
        return 1

    async def hincrby(self, key, field, amount=1):
        target = self.hashes.setdefault(key, {})
        target[str(field)] = str(int(target.get(str(field), 0)) + int(amount))
        return int(target[str(field)])

    async def zadd(self, key, mapping):
        target = self.zsets.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            added += int(str(member) not in target)
            target[str(member)] = float(score)
        return added

    async def zincrby(self, key, amount, member):
        target = self.zsets.setdefault(key, {})
        target[str(member)] = target.get(str(member), 0.0) + float(amount)
        return target[str(member)]

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

    async def zscore(self, key, member):
        return self.zsets.get(key, {}).get(str(member))

    async def zrevrange(self, key, start, end, withscores=False):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda item: (-item[1], item[0]))
        window = items[start:] if end == -1 else items[start:end + 1]
        if withscores:
            return [(member, score) for member, score in window]
        return [member for member, _ in window]

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    """FIFO replay of the queued commands, so the hash and zset end in the state MULTI/EXEC would leave."""

    def __init__(self, client: _FakeProfileRedis):
        self._client = client
        self._queued: list = []

    def _queue(self, command, *args, **kwargs):
        self._queued.append((command, args, kwargs))
        return self

    def get(self, *args, **kwargs):
        return self._queue(self._client.get, *args, **kwargs)

    def set(self, *args, **kwargs):
        return self._queue(self._client.set, *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._queue(self._client.delete, *args, **kwargs)

    def expire(self, *args, **kwargs):
        return self._queue(self._client.expire, *args, **kwargs)

    def hgetall(self, *args, **kwargs):
        return self._queue(self._client.hgetall, *args, **kwargs)

    def hset(self, *args, **kwargs):
        return self._queue(self._client.hset, *args, **kwargs)

    def hincrby(self, *args, **kwargs):
        return self._queue(self._client.hincrby, *args, **kwargs)

    def zadd(self, *args, **kwargs):
        return self._queue(self._client.zadd, *args, **kwargs)

    def zincrby(self, *args, **kwargs):
        return self._queue(self._client.zincrby, *args, **kwargs)

    def zcard(self, *args, **kwargs):
        return self._queue(self._client.zcard, *args, **kwargs)

    def zscore(self, *args, **kwargs):
        return self._queue(self._client.zscore, *args, **kwargs)

    async def execute(self):
        return [await command(*args, **kwargs) for command, args, kwargs in self._queued]


class _FakeRateRedis:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.ttls: dict[str, int | None] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.counts:
            return None
        self.counts[key] = int(value)
        self.ttls[key] = ex
        return True

    async def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


def _scored_point(pid: int, title: str) -> ScoredPoint:
    return ScoredPoint(
        id=pid,
        version=0,
        score=0.5,
        payload={
            "title": title,
            "url": f"https://example.com/{pid}",
            "published_date": "2026-01-01T00:00:00Z",
        },
        vector=None,
    )


_NEIGHBOUR_ID = 555


class _FakeQdrant:
    def __init__(self, known_ids):
        self.known_ids = set(known_ids)
        self.vector_queries: list[dict] = []

    async def retrieve(self, collection_name, ids, with_payload=False, with_vectors=False):
        return [SimpleNamespace(id=pid) for pid in ids if pid in self.known_ids]

    async def query_points(self, **kwargs):
        if "query" in kwargs:
            self.vector_queries.append(kwargs)
            return SimpleNamespace(points=[_scored_point(_NEIGHBOUR_ID, "vector hit")])
        return SimpleNamespace(points=[_scored_point(_NEIGHBOUR_ID, "category hit")])

    async def scroll(self, **kwargs):
        return ([_scored_point(_NEIGHBOUR_ID, "trending hit")], None)


def _via_local_proxy(app):
    """Pin the peer to loopback: the shipped default only honours X-Forwarded-For behind a loopback peer."""
    async def wrapper(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": ("127.0.0.1", 40000)}
        await app(scope, receive, send)

    return wrapper


_client = TestClient(_via_local_proxy(main.app), raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def profile_redis(monkeypatch):
    fake = _FakeProfileRedis()
    monkeypatch.setattr(user_profile, "_redis_client", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def rate_redis(monkeypatch):
    fake = _FakeRateRedis()
    monkeypatch.setattr(auth_module, "_rate_client", fake)
    return fake


@pytest.fixture
def generous_limits(monkeypatch):
    """Raise both limits so a shipped default cannot turn a state assertion into a 429."""
    monkeypatch.setattr(config, "PUBLIC_INTERACTION_RATE_PER_MIN", 10_000)
    monkeypatch.setattr(config, "INTERACTION_USER_RATE_PER_MIN", 10_000)


@pytest.fixture
def qdrant(monkeypatch):
    fake = _FakeQdrant(KNOWN_IDS)
    monkeypatch.setitem(main.state, "qdrant", fake)
    return fake


@pytest.fixture
def account(tmp_path, monkeypatch):
    store = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(store.connect())
    monkeypatch.setattr(auth_module, "store", store)

    def make(email: str):
        user = asyncio.run(store.create_user(email, "secret12", email.split("@")[0], "user"))
        token = asyncio.run(store.issue_token(user.id, 7))
        return user, auth_cookie(token)

    try:
        yield make
    finally:
        asyncio.run(store.close())


def _interact(cookie, article_id, kind="click"):
    return _client.post(
        "/recommend/interaction",
        json={"article_id": article_id, "interaction_type": kind},
        cookies=cookie,
    )


# The dicts hold the post-await state, so assertions need no event loop between HTTP calls.
def _hash(redis, key):
    return redis.hashes.get(key, {})


def _article_hash(redis, article_id):
    return _hash(redis, f"article:interactions:{article_id}")


def _detail_hash(redis, user_id, article_id):
    return _hash(redis, f"user:interaction_detail:{user_id}:{article_id}")


def _exists(redis, key):
    return key in redis.key_dump()


def _zset(redis, key):
    return redis.zsets.get(key, {})


def test_junk_interaction_types_never_become_hash_fields(
    account, profile_redis, qdrant, generous_limits
):
    """An unchecked kind would mint one permanent hash field per request; the seeded field set must stay byte-identical."""
    user, headers = account("junk-kind@example.com")
    article = KNOWN_IDS[0]

    seeded = _interact(headers, article, "click")
    assert seeded.status_code == 200, seeded.text
    before = set(_article_hash(profile_redis, article))
    assert before == {"click", "last_timestamp"}
    assert _detail_hash(profile_redis, user.id, article)["type"] == "click"

    statuses = [_interact(headers, article, _JUNK_KINDS[i % 2]).status_code for i in range(40)]
    assert set(statuses) == {422}, statuses

    after = set(_article_hash(profile_redis, article))
    assert len(after) == len(before), "the article hash grew from a junk interaction_type"
    assert after == before, after
    assert not after & {kind[:64] for kind in _JUNK_KINDS}
    # An accepted junk kind would have overwritten the stored type.
    assert _detail_hash(profile_redis, user.id, article)["type"] == "click"

    # The same guard for a caller that reaches record_interaction directly.
    for kind in _JUNK_KINDS:
        result = asyncio.run(
            user_profile.record_interaction(
                user_id=user.id, article_id=article, interaction_type=kind
            )
        )
        assert result is InteractionResult.INVALID_TYPE, kind
    assert set(_article_hash(profile_redis, article)) == before


def test_flood_of_distinct_kinds_stays_within_the_enum(
    account, profile_redis, qdrant, generous_limits
):
    user, headers = account("flood@example.com")
    flooded, saturated = KNOWN_IDS[0], KNOWN_IDS[1]

    assert _interact(headers, flooded, "click").status_code == 200
    flooded_fields = set(_article_hash(profile_redis, flooded))

    for kind in _JUNK_KINDS:
        _interact(headers, flooded, kind)
    assert set(_article_hash(profile_redis, flooded)) == flooded_fields

    # Same flood straight at the writer, for a caller that skips the body model.
    for kind in _JUNK_KINDS:
        result = asyncio.run(
            user_profile.record_interaction(
                user_id=user.id, article_id=flooded, interaction_type=kind
            )
        )
        assert result is InteractionResult.INVALID_TYPE, kind
    assert set(_article_hash(profile_redis, flooded)) == flooded_fields

    for kind in _LEGAL_KINDS:
        assert _interact(headers, saturated, kind).status_code == 200
    saturated_fields = set(_article_hash(profile_redis, saturated))
    assert saturated_fields == {*_LEGAL_KINDS, "last_timestamp"}
    assert len(saturated_fields) == len(InteractionType) + len(_BOOKKEEPING_FIELDS)
    assert saturated_fields <= set(_LEGAL_KINDS) | _BOOKKEEPING_FIELDS
    assert flooded_fields <= saturated_fields


def test_unknown_article_is_rejected_without_minting_keys(
    account, profile_redis, qdrant, generous_limits
):
    """A declined id must not even be cached: the rejection is keyed to a caller-chosen id, so a flood still grows the keyspace."""
    user, headers = account("unknown-article@example.com")
    article = KNOWN_IDS[0]

    declined = _interact(headers, UNKNOWN_ID)
    assert declined.status_code == 404
    assert declined.json()["detail"] == "Unknown article"

    assert not _exists(profile_redis, f"article:interactions:{UNKNOWN_ID}")
    assert not _exists(profile_redis, f"user:interaction_detail:{user.id}:{UNKNOWN_ID}")
    assert str(UNKNOWN_ID) not in _zset(profile_redis, f"user:interactions:{user.id}")
    assert not _zset(profile_redis, f"user:interactions:{user.id}")

    residue = {key for key in profile_redis.key_dump() if str(UNKNOWN_ID) in key}
    assert residue == set(), residue

    # The guard is not "reject everything": a real id in the same test is accepted.
    accepted = _interact(headers, article)
    assert accepted.status_code == 200
    assert accepted.json() == {"status": "ok", "article_id": article}
    assert _exists(profile_redis, f"article:interactions:{article}")
    assert _exists(profile_redis, f"user:interaction_detail:{user.id}:{article}")
    assert list(_zset(profile_redis, f"user:interactions:{user.id}")) == [str(article)]


def test_repeated_interactions_are_rate_limited_per_account(
    account, rate_redis, qdrant, monkeypatch
):
    """Only the per-account bucket is in reach, and a second account from the same peer still gets a full budget."""
    monkeypatch.setattr(config, "PUBLIC_INTERACTION_RATE_PER_MIN", 50)
    monkeypatch.setattr(config, "INTERACTION_USER_RATE_PER_MIN", 3)
    user, headers = account("flooder@example.com")
    article = KNOWN_IDS[0]

    statuses = [_interact(headers, article).status_code for _ in range(3 + 5)]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3:] == [429] * 5

    # Both axes counted; the per-IP bucket never reached its own limit.
    assert rate_redis.counts[f"user:rl:interaction:{user.id}"] == 8
    public_keys = [key for key in rate_redis.counts if key.startswith("public:rl:interaction:")]
    assert sum(rate_redis.counts[key] for key in public_keys) == 8
    assert all(rate_redis.counts[key] <= 50 for key in public_keys)

    # The per-IP axis is off, so nothing is charged to it.
    monkeypatch.setattr(config, "PUBLIC_INTERACTION_RATE_PER_MIN", 0)
    other, other_headers = account("bystander@example.com")
    assert other.id != user.id
    assert [_interact(other_headers, article).status_code for _ in range(3)] == [200, 200, 200]
    assert _interact(other_headers, article).status_code == 429
    # The first account's spent budget is unchanged: they did not share a bucket.
    assert rate_redis.counts[f"user:rl:interaction:{other.id}"] == 4
    assert rate_redis.counts[f"user:rl:interaction:{user.id}"] == 8
    assert sum(rate_redis.counts[key] for key in public_keys) == 8


def test_legitimate_click_is_recorded_and_reaches_the_recommender(
    account, profile_redis, qdrant, generous_limits
):
    """Asserted at every consumer: the counter, the detail record, get_user_interactions and the candidate generator itself."""
    user, headers = account("reader@example.com")
    article = KNOWN_IDS[0]

    response = _interact(headers, article, "click")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok", "article_id": article}

    assert _article_hash(profile_redis, article)["click"] == "1"
    detail = _detail_hash(profile_redis, user.id, article)
    assert detail["type"] == "click"
    assert detail["dwell_time_ms"] == "0"
    assert float(detail["timestamp"]) > 0

    assert asyncio.run(user_profile.get_trending_articles()) == [
        {"article_id": article, "score": 1.0}
    ]

    # The exact input the candidate generator reads.
    interactions = asyncio.run(user_profile.get_user_interactions(user.id))
    assert [article_id for article_id, _ in interactions] == [article]

    # The clicked article is the vector query and is excluded from what comes back.
    with patch.object(recommender, "state", {"qdrant": qdrant}):
        recommendations = asyncio.run(
            recommender.get_personalized_recommendations(user.id, limit=5)
        )
    excluded = {
        excluded_id
        for query in qdrant.vector_queries
        for condition in (query["query_filter"].must_not or [])
        for excluded_id in (condition.match.any if condition.match else ())
    }
    assert article in excluded
    assert [rec["id"] for rec in recommendations] == [_NEIGHBOUR_ID]


def test_distinct_interaction_cap_declines_new_article_but_allows_a_repeat(
    account, profile_redis, qdrant, generous_limits, monkeypatch
):
    """The cap bounds NEW key pairs, so a repeat of a seen article still records."""
    monkeypatch.setattr(config, "USER_MAX_DISTINCT_INTERACTIONS", 2)
    user, headers = account("capped@example.com")
    first, second, third = KNOWN_IDS

    assert _interact(headers, first).status_code == 200
    assert _interact(headers, second).status_code == 200

    declined = _interact(headers, third)
    assert declined.status_code == 429
    assert declined.json()["detail"] == "Interaction limit reached for this account"
    assert declined.headers["Retry-After"] == str(config.PUBLIC_RATE_WINDOW_SECONDS)
    assert not _exists(profile_redis, f"article:interactions:{third}")
    assert not _exists(profile_redis, f"user:interaction_detail:{user.id}:{third}")
    assert str(third) not in _zset(profile_redis, f"user:interactions:{user.id}")
    # The cap refuses the interaction, not the id's validity, so a positive cache entry is legitimate here.
    assert {key for key in profile_redis.key_dump() if str(third) in key} == {
        f"user_profile:article_exists:{third}"
    }
    assert profile_redis.strings[f"user_profile:article_exists:{third}"] == "1"

    assert _interact(headers, first).status_code == 200
    assert _article_hash(profile_redis, first)["click"] == "2"
    assert _detail_hash(profile_redis, user.id, first)["type"] == "click"
    assert {
        article_id for article_id, _ in asyncio.run(user_profile.get_user_interactions(user.id))
    } == {first, second}


def test_index_outage_does_not_poison_the_existence_cache(account, profile_redis, qdrant, generous_limits):
    """Caching the outcome of a failed lookup would latch "absent" for the whole TTL, 404ing real articles long after recovery."""
    _user, headers = account("outage@example.com")
    article = KNOWN_IDS[0]

    async def _explode(**kwargs):
        raise ConnectionError("qdrant down")

    qdrant.retrieve = _explode
    during = _interact(headers, article)
    # A failed lookup must not be reported as "unknown article", which reads as permanent.
    assert during.status_code == 503, during.text
    assert during.json()["detail"] == "Interaction store unavailable"
    assert not _exists(profile_redis, f"article:interactions:{article}")
    assert f"user_profile:article_exists:{article}" not in profile_redis.key_dump()

    # The index recovers; the next click must not replay a cached rejection.
    qdrant.retrieve = _FakeQdrant(KNOWN_IDS).retrieve
    after = _interact(headers, article)
    assert after.status_code == 200, after.text
    assert _article_hash(profile_redis, article)["click"] == "1"
    assert profile_redis.strings[f"user_profile:article_exists:{article}"] == "1"


# Pre-enum junk fields outlive any write-side fix (the article hash lives 90 days), so the read side needs its own allow-list.
def test_trending_ignores_legacy_junk_fields_seeded_before_the_enum(profile_redis):
    """Junk fields seeded before the enum must not sum in with the genuine clicks."""
    article = 101
    key = f"article:interactions:{article}"
    profile_redis.hashes[key] = {
        "click": "2",
        "view": "1",
        "last_timestamp": "1758000000.0",
    }
    for i in range(20):
        profile_redis.hashes[key][f"junk-{i}-" + "z" * 40] = "1"

    trending = asyncio.run(user_profile.get_trending_articles())

    assert trending == [{"article_id": article, "score": 3.0}], trending
    assert trending[0]["score"] != 23.0


def test_trending_excludes_the_last_timestamp_bookkeeping_field(profile_redis):
    """last_timestamp is a timestamp, not a counter; a naive sum would add ~1.7e9."""
    article = 202
    profile_redis.hashes[f"article:interactions:{article}"] = {
        "click": "4",
        "last_timestamp": "1758000000.0",
    }

    trending = asyncio.run(user_profile.get_trending_articles())

    assert trending == [{"article_id": article, "score": 4.0}], trending


def test_trending_article_with_only_junk_fields_does_not_rank(profile_redis):
    profile_redis.hashes["article:interactions:303"] = {
        f"junk-{i}": "50" for i in range(10)
    }

    assert asyncio.run(user_profile.get_trending_articles()) == []


# A kind the body model already rejected must answer INVALID_TYPE, not 404 "Unknown article".
def test_invalid_type_is_distinct_from_unknown_article(account, profile_redis, qdrant, generous_limits):
    from app.user_profile import record_interaction

    user, _headers = account("invalid-type@example.com")
    known = KNOWN_IDS[0]

    result = asyncio.run(
        record_interaction(user.id, known, interaction_type="TOTALLY-BOGUS")
    )

    assert result is InteractionResult.INVALID_TYPE, result
    assert result is not InteractionResult.UNKNOWN_ARTICLE
    assert not _exists(profile_redis, f"article:interactions:{known}")
    assert f"user_profile:article_exists:{known}" not in profile_redis.key_dump()


def test_known_article_with_invalid_type_is_not_404_over_http(account, profile_redis, qdrant, generous_limits):
    """404 would assert the article does not exist -- false for a real article."""
    _user, headers = account("invalid-type-http@example.com")
    known = KNOWN_IDS[0]

    response = _interact(headers, known, kind="TOTALLY-BOGUS")

    assert response.status_code == 422, response.text
    assert "Unknown article" not in response.text
    assert not _exists(profile_redis, f"article:interactions:{known}")


# A bucket keyed on the client IP still returns 429, so the per-account axis would silently collapse into a second per-IP axis.
def test_per_account_bucket_key_contains_the_subject_not_the_client_ip(monkeypatch):
    import asyncio as _asyncio

    from app import auth as auth_mod

    seen: list[str] = []

    class _RecordingRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            seen.append(key)
            return 1

    monkeypatch.setattr(auth_mod, "_rate_client", _RecordingRedis())

    class _Req:
        def __init__(self):
            self.headers = {}
            self.client = type("c", (), {"host": "10.0.0.5"})()
            self.state = type("s", (), {})()

    def _req_for(user_id):
        request = _Req()
        request.state.user_id = user_id
        return request

    for user_id in ("alice", "bob"):
        _asyncio.run(
            auth_mod._check_rate_limit(
                _req_for(user_id),
                "interaction",
                5,
                key_prefix="user:rl",
                window_seconds=60,
                fail_closed=True,
                subject=user_id,
            )
        )

    # Keying on the client IP would put both accounts in one budget.
    assert seen == ["user:rl:interaction:alice", "user:rl:interaction:bob"], seen
    assert seen[0] != seen[1]
    assert not any("10.0.0.5" in key for key in seen), seen
