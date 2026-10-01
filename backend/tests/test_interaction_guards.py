"""Guards on the Redis state /recommend/interaction mints per call.

Each call can mint one ``article:interactions:{id}`` hash FIELD per distinct
``interaction_type``, one ``user:interaction_detail:{user_id}:{id}`` key and one
member in ``user:interactions:{user_id}``; any of the three could be an
unbounded mint by one authenticated caller. Every guard is asserted against the
REAL route, the REAL ``require_auth`` dependency and the REAL Redis-backed rate
limiter -- only Redis itself is in memory -- and every "rejected" claim is
paired with evidence that no state was written.
"""

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

# Article ids the fake Qdrant index reports as present; anything else must be
# declined before a single key is written.
KNOWN_IDS = (101, 202, 303)
UNKNOWN_ID = 987654321

_LEGAL_KINDS = tuple(kind.value for kind in InteractionType)
_BOOKKEEPING_FIELDS = frozenset({"last_timestamp"})

# A long free-form kind and 60 further distinct ones: the shape of a loop that
# would mint one permanent hash field per iteration if the kind were unchecked.
_LONG_JUNK_KIND = "".join(
    random.Random(271).choice(string.ascii_letters + string.digits) for _ in range(8192)
)
_JUNK_KINDS = (
    "totally-made-up-kind",
    _LONG_JUNK_KIND,
) + tuple(f"junk-{i}-not-a-kind" for i in range(60))


class _FakeProfileRedis:
    """Minimal async Redis stand-in for the user-profile database.

    Only the commands the profile code issues are implemented; an unlisted one
    raises, so an unimplemented command surfaces as a test failure instead of
    becoming a no-op that fakes a passing guard.
    """

    def __init__(self):
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, int] = {}

    def key_dump(self) -> set[str]:
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
        # Read by the trending-index bootstrap to decide whether the one-time
        # backfill has already run.
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
        # Advances the trending index by the article's interaction total.
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
    """Queues the same commands as the real pipeline and applies them in order.

    ``record_interaction`` builds its whole write set in one pipeline, so FIFO
    replay leaves the hash fields and the sorted set in the state a real
    MULTI/EXEC would.
    """

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
    """The limiter's counter store: SET NX EX opens the window, then INCR."""

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
    """A minimal Qdrant point the recommender can score and format."""
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
    """Reports only ``known_ids`` from ``retrieve``; records candidate queries."""

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
    """Present requests as if they arrived via the loopback reverse proxy.

    Same wrapper as tests/test_main_http.py: the shipped default only honours
    X-Forwarded-For behind a loopback peer, and every request here comes from
    that one peer on purpose, so the per-IP axis behaves as it is deployed.
    """

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
    """Back the rate limiter with its own in-memory counter store.

    Rebuilt per test, so no counter leaks between cases; the route, its
    dependencies and the INCR/EXPIRE sequence are the shipped ones.
    """
    fake = _FakeRateRedis()
    monkeypatch.setattr(auth_module, "_rate_client", fake)
    return fake


@pytest.fixture
def generous_limits(monkeypatch):
    """Move both rate limits off the path for the tests that are not about them,
    so a shipped default can never turn a state assertion into a 429."""
    monkeypatch.setattr(config, "PUBLIC_INTERACTION_RATE_PER_MIN", 10_000)
    monkeypatch.setattr(config, "INTERACTION_USER_RATE_PER_MIN", 10_000)


@pytest.fixture
def qdrant(monkeypatch):
    """Install a Qdrant double that knows exactly which article ids exist."""
    fake = _FakeQdrant(KNOWN_IDS)
    monkeypatch.setitem(main.state, "qdrant", fake)
    return fake


@pytest.fixture
def account(tmp_path, monkeypatch):
    """Real accounts holding real session cookies from the real auth store."""
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


# The store's commands are coroutines because the app awaits them; the
# assertions read the same three dicts directly so a test can check what was
# written without borrowing an event loop between HTTP calls.


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
    """A non-enum kind must not add a field to the article hash.

    ``interaction_type`` is stored as the hash FIELD, so an unchecked string
    mints one permanent field per call. Over HTTP the body model rejects it
    (422) before the handler runs, and the guard behind it is checked directly
    on ``record_interaction``. Either way the seeded field set must survive
    byte-identical.
    """
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
    assert _detail_hash(profile_redis, user.id, article)["type"] == "click"

    # The guard itself, for a caller that reaches record_interaction directly
    # with a kind the body model never lets through.
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
    """Field growth is bounded by the enum, not by the number of requests.

    Two halves: the flood proves nothing new appears, and the three legal kinds
    pin the concrete maximum the hash is allowed to reach.
    """
    user, headers = account("flood@example.com")
    flooded, saturated = KNOWN_IDS[0], KNOWN_IDS[1]

    assert _interact(headers, flooded, "click").status_code == 200
    flooded_fields = set(_article_hash(profile_redis, flooded))

    for kind in _JUNK_KINDS:
        _interact(headers, flooded, kind)
    assert set(_article_hash(profile_redis, flooded)) == flooded_fields

    # The same flood straight at the writer, for a caller that skips the body
    # model: the field set still cannot move.
    for kind in _JUNK_KINDS:
        result = asyncio.run(
            user_profile.record_interaction(
                user_id=user.id, article_id=flooded, interaction_type=kind
            )
        )
        assert result is InteractionResult.INVALID_TYPE, kind
    assert set(_article_hash(profile_redis, flooded)) == flooded_fields

    # Concrete bound: one field per legal kind plus the timestamp bookkeeping
    # field -- no request count anywhere in it.
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
    """404 for an id that is not indexed, leaving Redis COMPLETELY untouched.

    A declined id must not even be remembered: caching it would key it to the
    caller-chosen id, so a flood would still grow the keyspace one key per
    probed id and an index blip would be latched "absent" for the whole TTL. So
    the residue check demands a key diff that is EMPTY, not merely small.
    """
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

    accepted = _interact(headers, article)
    assert accepted.status_code == 200
    assert accepted.json() == {"status": "ok", "article_id": article}
    assert _exists(profile_redis, f"article:interactions:{article}")
    assert _exists(profile_redis, f"user:interaction_detail:{user.id}:{article}")
    assert list(_zset(profile_redis, f"user:interactions:{user.id}")) == [str(article)]


def test_repeated_interactions_are_rate_limited_per_account(
    account, rate_redis, qdrant, monkeypatch
):
    """Real HTTP calls in a real loop: the surplus is answered 429.

    Phase 1 keeps the per-IP axis at a limit this call volume cannot reach, so
    the 429s can only come from the per-account bucket. Phase 2 disables the
    per-IP axis entirely and shows a second account, hitting the route from the
    SAME peer address, still gets its own full budget -- which is only possible
    if the bucket key carries the account id.
    """
    monkeypatch.setattr(config, "PUBLIC_INTERACTION_RATE_PER_MIN", 50)
    monkeypatch.setattr(config, "INTERACTION_USER_RATE_PER_MIN", 3)
    user, headers = account("flooder@example.com")
    article = KNOWN_IDS[0]

    statuses = [_interact(headers, article).status_code for _ in range(3 + 5)]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3:] == [429] * 5

    # Both axes counted, and the per-IP counter never reached its own limit, so
    # it is not what answered 429.
    assert rate_redis.counts[f"user:rl:interaction:{user.id}"] == 8
    public_keys = [key for key in rate_redis.counts if key.startswith("public:rl:interaction:")]
    assert sum(rate_redis.counts[key] for key in public_keys) == 8
    assert all(rate_redis.counts[key] <= 50 for key in public_keys)

    monkeypatch.setattr(config, "PUBLIC_INTERACTION_RATE_PER_MIN", 0)
    other, other_headers = account("bystander@example.com")
    assert other.id != user.id
    assert [_interact(other_headers, article).status_code for _ in range(3)] == [200, 200, 200]
    assert _interact(other_headers, article).status_code == 429
    assert rate_redis.counts[f"user:rl:interaction:{other.id}"] == 4
    assert rate_redis.counts[f"user:rl:interaction:{user.id}"] == 8
    assert sum(rate_redis.counts[key] for key in public_keys) == 8


def test_legitimate_click_is_recorded_and_reaches_the_recommender(
    account, profile_redis, qdrant, generous_limits
):
    """A real click on a real article: recorded, counted, and consumed.

    Asserted at every consumer the app actually uses: the per-article counter,
    the per-user detail record, ``get_user_interactions`` (the exact input
    ``get_personalized_recommendations`` builds its candidate queries from) and
    that function itself, driven end to end over the same fake Redis and Qdrant.
    """
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

    interactions = asyncio.run(user_profile.get_user_interactions(user.id))
    assert [article_id for article_id, _ in interactions] == [article]

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
    """The cap stops new articles; re-visiting a known one still records.

    Re-interacting must keep working: the cap is about minting NEW key pairs,
    and a returning reader is the documented, expected case.
    """
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
    # `third` IS indexed, so the existence cache legitimately holds a POSITIVE
    # entry for it: the cap refuses the interaction, not the id's validity.
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
    """A Qdrant failure must not make real articles 404 for the cache's TTL.

    Caching the outcome of a FAILED lookup would latch "absent" for the whole
    300s TTL, so a momentary index blip would keep rejecting genuine articles
    long after the index recovered. This drives a real outage and a real
    recovery through the endpoint.
    """
    _user, headers = account("outage@example.com")
    article = KNOWN_IDS[0]

    async def _explode(**kwargs):
        raise ConnectionError("qdrant down")

    qdrant.retrieve = _explode
    during = _interact(headers, article)
    # The index could not answer, so the endpoint must say so rather than claim
    # the article is unknown, which is a different, permanent-sounding statement.
    assert during.status_code == 503, during.text
    assert during.json()["detail"] == "Interaction store unavailable"
    assert not _exists(profile_redis, f"article:interactions:{article}")
    assert f"user_profile:article_exists:{article}" not in profile_redis.key_dump()

    qdrant.retrieve = _FakeQdrant(KNOWN_IDS).retrieve
    after = _interact(headers, article)
    assert after.status_code == 200, after.text
    assert _article_hash(profile_redis, article)["click"] == "1"
    assert profile_redis.strings[f"user_profile:article_exists:{article}"] == "1"


# The write-side enum stops new junk, but article:interactions:{id} keys live for
# USER_INTERACTION_TTL_DAYS (90 by default), so a field an older build minted is
# still in the hash -- and a name-blind trending sum counts those fields too,
# letting an attacker who poisoned the hash keep inflating a chosen article's
# score for the whole TTL. A write-only fix would leave that exposure open.


def test_trending_ignores_legacy_junk_fields_seeded_before_the_enum(profile_redis):
    """Only the known kinds contribute to an article's trending score.

    The hash is seeded the shape an attacker would leave behind: many junk fields
    plus a couple of genuine clicks. Without a read-side allow-list the junk sums
    in and the article scores far higher than it earned.
    """
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
    """last_timestamp is not a counter and must never add to the score.

    It is a float-ish string, so a naive sum that accepted it would silently
    add a huge number to every article.
    """
    article = 202
    profile_redis.hashes[f"article:interactions:{article}"] = {
        "click": "4",
        "last_timestamp": "1758000000.0",
    }

    trending = asyncio.run(user_profile.get_trending_articles())

    assert trending == [{"article_id": article, "score": 4.0}], trending


def test_trending_article_with_only_junk_fields_does_not_rank(profile_redis):
    """An article poisoned only with junk drops out of trending entirely."""
    profile_redis.hashes["article:interactions:303"] = {
        f"junk-{i}": "50" for i in range(10)
    }

    assert asyncio.run(user_profile.get_trending_articles()) == []


# A rejected kind must not report itself as an unindexed id: the route would then
# answer 404 "Unknown article" for an article that demonstrably exists. Over HTTP
# the pydantic model rejects the kind first with 422, so only a direct caller
# reaches the write layer -- the last line of defence, and it must not lie either.


def test_invalid_type_is_distinct_from_unknown_article(account, profile_redis, qdrant, generous_limits):
    """A junk kind on a KNOWN article yields INVALID_TYPE, not UNKNOWN_ARTICLE."""
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
    """End to end: a junk kind on a real article is 422, never 404.

    404 would assert the article does not exist; 422 says the request body was
    invalid, which is the truth.
    """
    _user, headers = account("invalid-type-http@example.com")
    known = KNOWN_IDS[0]

    response = _interact(headers, known, kind="TOTALLY-BOGUS")

    assert response.status_code == 422, response.text
    assert "Unknown article" not in response.text
    assert not _exists(profile_redis, f"article:interactions:{known}")


# This is the property a careless merge silently loses: if the key line keeps
# only the client IP, user_rate_limit still works, still returns 429, and no
# test of the ROUTE fails -- the per-account axis has silently become a second
# per-IP axis, and a distributed flood is unbounded again.


def test_per_account_bucket_key_contains_the_subject_not_the_client_ip(monkeypatch):
    """Two subjects behind one IP must occupy two DIFFERENT buckets."""
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

    assert seen == ["user:rl:interaction:alice", "user:rl:interaction:bob"], seen
    assert seen[0] != seen[1]
    assert not any("10.0.0.5" in key for key in seen), seen

