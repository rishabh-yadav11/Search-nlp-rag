"""In-memory fakes for the external services the backend talks to.

These deliberately fake ONLY the four boundary objects the real app creates:
an AsyncQdrantClient, an AsyncOpenAI LLM client, and redis.asyncio clients
(every module builds them through ``redis.asyncio.from_url`` — cache, health,
auth rate limiter, analytics, cost_budget, user_profile). Everything else runs
for real: routers, middleware, stores, pydantic models.

The fakes are self-contained (no ``app.*`` imports) so they can live in one
module without circular-import surprises.

* :class:`FakeQdrant` — an in-memory Qdrant over a tiny article list.
  Supports the search path (``query_points`` with ``prefetch`` + Fusion),
  body/point retrieval (``retrieve``), paginated scrolling (``scroll``) with
  an optional ``order_by``, collection checks (``collection_exists``) and
  enough filter semantics (MatchAny / DatetimeRange conditions) to make the
  hard date-window tests meaningful.
* :class:`FakeLLMClient` — AsyncOpenAI stand-in with a canned, deterministic
  answer (and an SSE stream variant).
* :class:`FakeRedis` — a small in-memory Redis covering the command surface
  the app actually uses (strings, hashes, sorted sets, exists/expire/scan,
  pipelines). ``redis.asyncio.from_url`` is monkeypatched to return it.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
from datetime import UTC, datetime


class FakePoint:
    """A qdrant ScoredPoint/Record substitute: id + payload + score."""

    def __init__(self, point_id: int, payload: dict | None = None, score: float = 0.5):
        self.id = point_id
        self.payload = payload if payload is not None else {}
        self.score = score


class FakeQueryResponse:
    def __init__(self, points: list[FakePoint]):
        self.points = points


def seed_articles() -> dict[int, dict]:
    """A tiny, assertable corpus spanning several years and facets.

    Dates stay strictly ISO ``YYYY-MM-DD`` so the fake's date comparisons and
    the app's ``_parse_date`` agree. ``body`` is short but present (the chat
    path attaches bodies).
    """
    return {
        1: {
            "id": 1,
            "title": "PhonePe raises $100M funding round led by General Atlantic",
            "url": "https://example.test/phonepe-100m",
            "published_date": "2025-06-15",
            "category": "Venture Capital",
            "summary": "PhonePe closed a $100M funding round from General Atlantic.",
            "body": "PhonePe, the payments firm, closed a $100 million funding round led by General Atlantic.",
            "author_names": ["Alice Rao"],
            "industry_names": ["Finance"],
            "dealtype_names": ["Venture Capital"],
            "tag_names": ["funding", "fintech", "payments"],
            "content_type": "article",
        },
        2: {
            "id": 2,
            "title": "Ola Electric files draft papers for mega IPO",
            "url": "https://example.test/ola-ipo",
            "published_date": "2025-03-02",
            "category": "IPO",
            "summary": "Ola Electric filed draft papers for its IPO.",
            "body": "Ola Electric filed draft red herring prospectus for its planned initial public offering.",
            "author_names": ["Bob Sen"],
            "industry_names": ["Mobility"],
            "dealtype_names": ["IPO"],
            "tag_names": ["ipo", "electric vehicles"],
            "content_type": "article",
        },
        3: {
            "id": 3,
            "title": "Swiggy acquires restaurant-tech startup for $400M",
            "url": "https://example.test/swiggy-acquires",
            "published_date": "2024-11-21",
            "category": "M&A",
            "summary": "Swiggy acquired a restaurant-tech startup in a $400M deal.",
            "body": "Swiggy bought Dyno Foods, a restaurant-tech startup, in a $400 million acquisition.",
            "author_names": ["Carol D'Souza"],
            "industry_names": ["Consumer"],
            "dealtype_names": ["M&A"],
            "tag_names": ["acquisition", "food delivery", "consumer"],
            "content_type": "article",
        },
        4: {
            "id": 4,
            "title": "Zomato posts wider losses in Q1 2025",
            "url": "https://example.test/zomato-q1",
            "published_date": "2025-08-01",
            "category": "Earnings",
            "summary": "Zomato reported wider quarterly losses.",
            "body": "Zomato reported a wider net loss for Q1 2025 amid rising food delivery costs.",
            "author_names": ["Alice Rao"],
            "industry_names": ["Consumer"],
            "dealtype_names": ["Earnings"],
            "tag_names": ["earnings", "food delivery"],
            "content_type": "article",
        },
        5: {
            "id": 5,
            "title": "Fintech startup raises early-stage round in Mumbai",
            "url": "https://example.test/fintech-early",
            "published_date": "2021-05-10",
            "category": "Venture Capital",
            "summary": "A fintech startup raised a seed round in Mumbai.",
            "body": "A small fintech startup raised an early-stage funding round in Mumbai.",
            "author_names": ["Dave Iyer"],
            "industry_names": ["Finance"],
            "dealtype_names": ["Venture Capital"],
            "tag_names": ["funding", "fintech"],
            "content_type": "article",
        },
        6: {
            "id": 6,
            "title": "Old economy conglomerate sells stake in unit",
            "url": "https://example.test/stake-sale",
            "published_date": "2019-12-20",
            "category": "Stake Sale",
            "summary": "An old-economy conglomerate sold a stake in a subsidiary.",
            "body": "The conglomerate sold a minority stake in its unit to a private investor.",
            "author_names": ["Eve Rao"],
            "industry_names": ["Industrials"],
            "dealtype_names": ["Stake Sale"],
            "tag_names": ["stake sale"],
            "content_type": "article",
        },
    }


def _coerce_dt(value: object) -> datetime | None:
    """Parse a stored published_date (ISO date, or datetime string) to UTC aware."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if dt.tzinfo is None:

        dt = dt.replace(tzinfo=UTC)
    return dt


class FakeQdrant:
    """In-memory AsyncQdrantClient substitute over a dict of article dicts."""

    def __init__(self, articles: dict[int, dict] | None = None):
        self.articles = dict(articles) if articles is not None else seed_articles()
        self.collection_prefix = "vccircle"

    # -- helpers -----------------------------------------------------------

    def _payload_match_value(self, cond, payload: dict) -> bool:
        """Does ``payload`` satisfy one FieldCondition's match/range?."""
        from qdrant_client.models import MatchValue

        match = getattr(cond, "match", None)
        rng = getattr(cond, "range", None)
        key = getattr(cond, "key", None)
        if match is not None:
            stored = payload.get(key)
            if isinstance(match, MatchValue):
                wanted = {match.value}
            else:
                # MatchAny and everything else expose .any (a set/list)
                wanted = set(getattr(match, "any", ()) or ())
            if isinstance(stored, list):
                haystack = {str(v) for v in stored}
            else:
                haystack = {str(stored)} if stored is not None else set()
            return bool(wanted & haystack)
        if rng is not None:
            stored_dt = _coerce_dt(payload.get(key))
            if stored_dt is None:
                return False
            # DatetimeRange exposes gte/lte/gt/lt as attributes.
            for attr, op in (("gte", ">="), ("lte", "<="), ("gt", ">"), ("lt", "<")):
                bound = getattr(rng, attr, None)
                if bound is None:
                    continue
                bound_dt = _coerce_dt(bound)
                if bound_dt is None:
                    continue
                if op == ">=" and not stored_dt >= bound_dt:
                    return False
                if op == "<=" and not stored_dt <= bound_dt:
                    return False
                if op == ">" and not stored_dt > bound_dt:
                    return False
                if op == "<" and not stored_dt < bound_dt:
                    return False
            return True
        return True

    def _filter_records(self, qfilter) -> list[FakePoint]:
        """Apply a Qdrant ``Filter`` (must / should / must_not) to the corpus."""
        points = [FakePoint(a["id"], dict(a), score=a.get("_score", 0.9)) for a in self.articles.values()]
        if qfilter is None:
            return points
        must = getattr(qfilter, "must", None) or []
        must_not = getattr(qfilter, "must_not", None) or []
        # Qdrant ``should`` semantics: a non-empty list is an OR — the point
        # must match at least one should-condition. An empty/absent list is no
        # constraint at all (it never *requires* a match).
        should = getattr(qfilter, "should", None) or []
        out = []
        for p in points:
            if not all(self._payload_match_value(c, p.payload) for c in must):
                continue
            if any(self._payload_match_value(c, p.payload) for c in must_not):
                continue
            if should and not any(self._payload_match_value(c, p.payload) for c in should):
                continue
            out.append(p)
        return out

    @staticmethod
    def _select_payload(point: FakePoint, with_payload: object) -> FakePoint:
        if with_payload is True:
            return point
        if with_payload in (False, None):
            return FakePoint(point.id, {}, point.score)
        keys = set(with_payload or [])
        if "body" not in keys:
            # list request carries only what the caller asked for; keep body only
            # when explicitly wanted
            keys = keys - {"body"}
        payload = {k: v for k, v in (point.payload or {}).items() if k in keys}
        return FakePoint(point.id, payload, point.score)

    # -- qdrant-client API surface used by the app --------------------------

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name.startswith(self.collection_prefix) or collection_name == "vccircle_articles"

    async def query_points(self, **kwargs: object) -> FakeQueryResponse:
        qfilter = kwargs.get("query_filter")
        limit = int(kwargs.get("limit", 10))
        with_payload = kwargs.get("with_payload", True)
        points = self._filter_records(qfilter)
        query = kwargs.get("query")
        # A point-id query (recommender similar/for-you) excludes the id itself.
        if isinstance(query, int):
            points = [p for p in points if p.id != query]
        # Deterministic ordering: probe-first so repeated calls agree.
        points.sort(key=lambda p: (p.score, p.id), reverse=True)
        return FakeQueryResponse([
            self._select_payload(p, with_payload) for p in points[:limit]
        ])

    async def retrieve(self, **kwargs: object) -> list[FakePoint]:
        ids = kwargs.get("ids")
        point_id = kwargs.get("point_id")
        with_payload = kwargs.get("with_payload", True)
        if ids is not None:
            wanted = {int(i) for i in ids}
        elif point_id is not None:
            wanted = {int(point_id)}
        else:
            wanted = set()
        result = []
        for a in self.articles.values():
            if int(a["id"]) not in wanted:
                continue
            point = FakePoint(a["id"], dict(a), score=a.get("_score", 0.9))
            result.append(self._select_payload(point, with_payload))
        return result

    async def scroll(self, **kwargs: object) -> tuple[list[FakePoint], None]:
        qfilter = kwargs.get("scroll_filter")
        limit = int(kwargs.get("limit", 10))
        with_payload = kwargs.get("with_payload", True)
        order_by = kwargs.get("order_by")
        points = self._filter_records(qfilter)
        if order_by and order_by.get("key") == "published_date":
            points.sort(
                key=lambda p: _coerce_dt((p.payload or {}).get("published_date"))
                or datetime.min.replace(tzinfo=UTC),
                reverse=(order_by.get("direction", "desc") == "desc"),
            )
        else:
            points.sort(key=lambda p: p.id)
        return [self._select_payload(p, with_payload) for p in points[:limit]], None

    async def count(self, **kwargs: object) -> int:
        qfilter = kwargs.get("query_filter")
        return len(self._filter_records(qfilter))

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

CANONICAL_ANSWER = (
    "Ola Electric filed draft papers for its mega IPO, and PhonePe closed a "
    "$100M funding round. The articles that cover this are [1][2]."
)


class _FakeAssistant:
    content: str

    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, delta_content: str | None = None):
        self.message = _FakeAssistant(content)
        self.delta = type("Delta", (), {"content": delta_content})()


class _FakeUsage:
    prompt_tokens = 12
    completion_tokens = 9


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeStreamChunk:
    """One SSE chunk: choices with a delta, plus an optional usage trailer."""

    def __init__(self, text: str, usage: bool = False):
        self.choices = [_FakeChoice("", delta_content=text)] if text else []
        self.usage = _FakeUsage() if usage else None


async def _stream_chunks():
    for word in CANONICAL_ANSWER.split(" "):
        yield _FakeStreamChunk(word + " ")
    yield _FakeStreamChunk("", usage=True)


class _FakeCompletions:
    def __init__(self, owner: FakeLLMClient):
        self._owner = owner

    async def create(self, **kwargs: object) -> _FakeResponse:
        asyncio.create_task  # noqa: B018 - imported for symmetry; no-op here
        if kwargs.get("stream"):
            return _stream_chunks()
        return _FakeResponse(self._owner.answer)


class _FakeChat:
    def __init__(self, owner: FakeLLMClient):
        self.completions = _FakeCompletions(owner)


# ---------------------------------------------------------------------------
# Fake model objects (the lifespan builds DenseEncoder / SparseTextEmbedding /
# Reranker; they are replaced so no model weights are downloaded or loaded)
# ---------------------------------------------------------------------------

class _Vector:
    """Thin numpy-like object exposing the ``.tolist()`` the app calls."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self) -> list:
        return list(self._values)


class FakeDenseEncoder:
    """Stand-in for ``app.encoders.DenseEncoder``: deterministic unit vector."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def encode(self, text: str) -> _Vector:
        del text
        # A fixed-length vector; only its tolist() form is ever consumed.
        return _Vector([0.1, 0.2, 0.3, 0.4, 0.5])


class FakeSparseVec:
    """Sparse embedding stand-in with numpy-like indices/values."""

    def __init__(self):
        self.indices = _Vector([0, 1, 2, 3])
        self.values = _Vector([0.5, 0.25, 0.125, 0.0625])


class FakeSparseModel:
    """Stand-in for ``fastembed.SparseTextEmbedding``: lazy embed generator."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def embed(self, texts: list[str]):
        del texts

        def _gen():
            yield FakeSparseVec()

        return _gen()


class FakeReranker:
    """Stand-in for ``app.reranker.Reranker``: deterministic positive logits.

    Sigmoid-normalized these land around 0.73, comfortably above the chat
    relevance gate (ASK_MIN_SCORE=0.2), so a chat turn reaches the LLM with
    real sources instead of the weak-results fallback.
    """

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.backend = "fake"

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0] * len(pairs)


class FakeLLMClient:
    """Minimal AsyncOpenAI substitute with a canned answer.

    The real AsyncOpenAI requires a live network only on ``create``; the object
    construction itself is cheap. ``app.main`` monkeypatches ``AsyncOpenAI`` to
    return this, and the lifespan keeps calling the constructor with whatever
    kwargs it is given.
    """

    def __init__(self, *args: object, **kwargs: object):
        del args, kwargs
        self.answer = CANONICAL_ANSWER
        self.chat = _FakeChat(self)


# ---------------------------------------------------------------------------
# Fake Redis
# ---------------------------------------------------------------------------

class _FakePipeline:
    """Collects commands, runs them at execute() in order."""

    def __init__(self, owner: FakeRedis):
        self._owner = owner
        self._ops: list[tuple[tuple, dict]] = []

    def _queue(self, command, *args, **kwargs):
        self._ops.append((command, (args, kwargs)))

    def get(self, key):
        self._queue("get", key)
        return self

    def set(self, key, value, **kwargs):
        self._queue("set", key, value, **kwargs)
        return self

    def delete(self, *keys):
        self._queue("delete", *keys)
        return self

    def hgetall(self, key):
        self._queue("hgetall", key)
        return self

    def hget(self, key, field):
        self._queue("hget", key, field)
        return self

    def hset(self, key, *args, **kwargs):
        self._queue("hset", key, *args, **kwargs)
        return self

    def hincrby(self, key, field, amount=1):
        self._queue("hincrby", key, field, amount)
        return self

    def zcard(self, key):
        self._queue("zcard", key)
        return self

    def zscore(self, key, member):
        self._queue("zscore", key, member)
        return self

    def zadd(self, key, mapping):
        self._queue("zadd", key, mapping)
        return self

    def zincrby(self, key, amount, member):
        self._queue("zincrby", key, amount, member)
        return self

    def incr(self, key, amount=1):
        self._queue("incr", key, amount)
        return self

    def expire(self, key, ttl):
        self._queue("expire", key, ttl)
        return self

    async def execute(self) -> list:
        results = []
        for command, (args, kwargs) in self._ops:
            fn = getattr(self._owner, command)
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            results.append(result)
        return results


class FakeRedis:
    """In-memory Redis supporting the subset of commands the app uses.

    All values stored as strings (decode_responses=True semantics). Shares one
    keyspace across every ``from_url`` caller so cross-module reads see writes.
    """

    def __init__(self):
        self._data: dict[str, str] = {}

    def reset(self) -> None:
        self._data.clear()

    # strings
    async def get(self, key):
        return self._data.get(key)

    async def set(self, key, value, ex=None, nx=False):
        del ex  # TTLs are irrelevant in-memory; kept for interface parity
        if nx and key in self._data:
            return None
        self._data[key] = (
            str(value) if not isinstance(value, (dict, list, tuple)) else json.dumps(value)
        )
        return True

    async def mget(self, keys):
        return [self._data.get(k) for k in keys]

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                n += 1
        return n

    async def incr(self, key, amount=1):
        cur = int(self._data.get(key, "0"))
        cur += amount
        self._data[key] = str(cur)
        return cur

    async def exists(self, key):
        return 1 if key in self._data else 0

    async def expire(self, key, ttl):
        del ttl
        return 1 if key in self._data else 0

    async def scan(self, cursor=0, match=None, count=None):
        del count
        keys = []
        for k in self._data:
            if match is None or fnmatch.fnmatch(k, match):
                keys.append(k)
        return (0, keys)

    # hashes
    async def hset(self, key, *args, **kwargs):
        mapping = kwargs.get("mapping")
        if mapping is not None:
            pairs = list(mapping.items())
        else:
            pairs = list(zip(args[::2], args[1::2]))
        store = json.loads(self._data.get(f"__h:{key}", "{}"))
        for field, value in pairs:
            store[str(field)] = str(value)
        self._data[f"__h:{key}"] = json.dumps(store)
        return len(pairs)

    async def hget(self, key, field):
        store = json.loads(self._data.get(f"__h:{key}", "{}"))
        return store.get(str(field))

    async def hgetall(self, key):
        store = json.loads(self._data.get(f"__h:{key}", "{}"))
        return dict(store)

    async def hincrby(self, key, field, amount=1):
        store = json.loads(self._data.get(f"__h:{key}", "{}"))
        cur = int(store.get(str(field), "0")) + int(amount)
        store[str(field)] = str(cur)
        self._data[f"__h:{key}"] = json.dumps(store)
        return cur

    async def hdel(self, key, *fields):
        store = json.loads(self._data.get(f"__h:{key}", "{}"))
        n = 0
        for f in fields:
            if str(f) in store:
                del store[str(f)]
                n += 1
        self._data[f"__h:{key}"] = json.dumps(store)
        return n

    # sorted sets
    def _zset(self, key):
        data = {}
        raw = self._data.get(f"__z:{key}")
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
        return data

    def _write_zset(self, key, data):
        self._data[f"__z:{key}"] = json.dumps(data)

    async def zadd(self, key, mapping):
        data = self._zset(key)
        for member, score in mapping.items():
            data[str(member)] = float(score)
        self._write_zset(key, data)
        return len(mapping)

    async def zincrby(self, key, amount, member):
        data = self._zset(key)
        data[str(member)] = float(data.get(str(member), 0.0)) + float(amount)
        self._write_zset(key, data)
        return float(data[str(member)])

    async def zcard(self, key):
        return len(self._zset(key))

    async def zscore(self, key, member):
        return self._zset(key).get(str(member))

    def _zitems(self, key):
        return sorted(self._zset(key).items(), key=lambda kv: (kv[1], kv[0]))

    def _zmembers(self, key, reverse=False):
        return sorted(self._zset(key).items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)

    async def zrange(self, key, start, stop, withscores=False):
        items = self._zmembers(key)[start:stop]
        if withscores:
            return [(member, score) for member, score in items]
        return [member for member, _ in items]

    async def zrevrange(self, key, start, stop, withscores=False):
        items = self._zmembers(key, reverse=True)[start:stop]
        if withscores:
            return [(member, score) for member, score in items]
        return [member for member, _ in items]

    async def zrem(self, key, *members):
        data = self._zset(key)
        n = 0
        for m in members:
            if str(m) in data:
                del data[str(m)]
                n += 1
        self._write_zset(key, data)
        return n

    # generic
    def pipeline(self):
        return _FakePipeline(self)

    async def ping(self):
        return True

    async def aclose(self):
        return None


class FakeRedisFactory:
    """Callable standing in for ``redis.asyncio.from_url``; returns a shared fake."""

    def __init__(self):
        self.instance = FakeRedis()

    def __call__(self, *args, **kwargs):
        del args, kwargs
        return self.instance
