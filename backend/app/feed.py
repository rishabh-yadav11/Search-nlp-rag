"""Personalized feed: per-user subscriptions to tag/industry/dealtype values,
surfaced as a recency-ordered article feed matching ANY subscription.

Subscriptions live in their own SQLite file (``user_subscriptions``), keyed by
(user_id, kind, value) so re-adding an existing subscription is a no-op
(``INSERT OR IGNORE``) rather than an error. The feed itself is a plain Qdrant
scroll ordered by ``published_date`` desc, with a ``should`` Filter built from
the subscription values — one FieldCondition per kind that has subscribed
values (tag/industry/dealtype map to payload tag_names/industry_names/
dealtype_names respectively).

The store mirrors ChatStore's discipline: WAL journaling, busy_timeout, foreign
keys enabled, and a ``_unit_of_work`` context manager for atomic multi-statement
writes. Reads use the shared connection.
"""

import logging
import os
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, field_validator
from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.auth import require_auth, require_permission
from app.config import config
from app.input_hygiene import MAX_FACET_VALUE_LEN, normalize_text

logger = logging.getLogger("feed")

router = APIRouter(
    prefix="/api/feed",
    tags=["feed"],
    dependencies=[Depends(require_auth), Depends(require_permission("feed:use"))],
)

# Module-level store; set by main.lifespan (and by tests).
store: "FeedStore | None" = None

VALID_KINDS = ("tag", "industry", "dealtype")
# kind -> payload field name carrying the subscribed facet values.
KIND_FACET_KEY = {
    "tag": "tag_names",
    "industry": "industry_names",
    "dealtype": "dealtype_names",
}


class SubscriptionOut(BaseModel):
    kind: str
    value: str
    created_at: float


class SubscriptionIn(BaseModel):
    kind: str
    value: str

    @field_validator("kind")
    @classmethod
    def _kind_valid(cls, v: str) -> str:
        if v not in VALID_KINDS:
            raise ValueError("kind must be one of tag|industry|dealtype")
        return v

    @field_validator("value")
    @classmethod
    def _value_valid(cls, v: str) -> str:
        v = normalize_text(v)
        if not v:
            raise ValueError("value must not be empty")
        if len(v) > MAX_FACET_VALUE_LEN:
            raise ValueError(f"value too long (maximum {MAX_FACET_VALUE_LEN} characters)")
        return v


class AddResponse(BaseModel):
    ok: bool
    added: bool


class RemoveResponse(BaseModel):
    ok: bool
    removed: bool


class FeedSummary(BaseModel):
    """Same shape as main.SourceSummary; defined here to avoid a circular import
    (feed is imported by main before SourceSummary exists). The feed has no real
    scores, so this always carries score=0.0."""

    id: int
    title: str
    url: str
    published_date: str | None = None
    category: str | None = None
    summary: str = ""
    author_names: list[str] = []
    industry_names: list[str] = []
    dealtype_names: list[str] = []
    tag_names: list[str] = []
    content_type: str | None = None
    score: float


class FeedResponse(BaseModel):
    results: list[FeedSummary]
    note: str | None = None


class SubscriptionsResponse(BaseModel):
    subscriptions: list[SubscriptionOut]


EMPTY_SUBSCRIPTIONS_NOTE = "Subscribe to tags, industries or deal types to see your feed."
EMPTY_RESULTS_NOTE = "No articles match your subscriptions yet."


def _now() -> float:
    return time.time()


def _group_by_kind(subscriptions: Sequence[SubscriptionOut]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for sub in subscriptions:
        grouped.setdefault(sub.kind, []).append(sub.value)
    return grouped


def _point_to_summary(point) -> FeedSummary:
    """Build a FeedSummary (SourceSummary-shaped) from one scroll point."""
    payload = point.payload or {}
    return FeedSummary(
        id=point.id,
        title=payload.get("title", ""),
        url=payload.get("url", ""),
        published_date=payload.get("published_date"),
        category=payload.get("category"),
        summary=payload.get("summary", ""),
        author_names=payload.get("author_names") or [],
        industry_names=payload.get("industry_names") or [],
        dealtype_names=payload.get("dealtype_names") or [],
        tag_names=payload.get("tag_names") or [],
        content_type=payload.get("content_type"),
        score=0.0,
    )


class FeedStore:
    """SQLite store of user subscriptions.

    WAL mode + busy_timeout so multiple gunicorn workers can read/write
    concurrently without "database is locked". Foreign keys are always enabled,
    both on the shared connection and inside every unit of work.
    """

    def __init__(self, path: str):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        # Idempotent: a second call must not clobber the live connection, which
        # would abandon its aiosqlite worker thread (non-daemon) and hang
        # interpreter shutdown.
        if self._db is not None:
            return
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (user_id, kind, value)
            )
            """
        )

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("FeedStore is not connected; call connect() first")
        return self._db

    @asynccontextmanager
    async def _unit_of_work(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run a group of statements as ONE atomic unit on a dedicated connection.

        Mirrors ChatStore._unit_of_work: ``isolation_level=None`` keeps the
        explicit BEGIN from nesting inside an implicit one, ``BEGIN IMMEDIATE``
        takes the WAL write lock up front so a mid-transaction lock upgrade
        cannot fail past the busy timeout, and ``foreign_keys`` is per-connection
        so it must be set here too.
        """
        db = await aiosqlite.connect(self._path, isolation_level=None)
        try:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute("BEGIN IMMEDIATE")
            yield db
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def list_subscriptions(self, user_id: str) -> list[SubscriptionOut]:
        db = self._require_db()
        rows = await db.execute_fetchall(
            "SELECT kind, value, created_at FROM user_subscriptions"
            " WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        )
        return [
            SubscriptionOut(kind=r["kind"], value=r["value"], created_at=r["created_at"])
            for r in rows
        ]

    async def add_subscription(self, user_id: str, kind: str, value: str) -> bool:
        """Insert (or no-op on duplicate). Returns True if actually added.

        Ran in ONE unit of work so the cap CHECK and the INSERT share a single
        write transaction (a dedicated connection could not see the other's lock):
        a re-add of an existing subscription returns added=False and never counts
        against the cap.
        """
        ts = _now()
        async with self._unit_of_work() as db:
            count_rows = await db.execute_fetchall(
                "SELECT COUNT(*) AS n FROM user_subscriptions WHERE user_id = ?",
                (user_id,),
            )
            count = count_rows[0]["n"] if count_rows else 0
            existing = await db.execute_fetchall(
                "SELECT 1 FROM user_subscriptions WHERE user_id = ? AND kind = ? AND value = ?",
                (user_id, kind, value),
            )
            if not existing and count >= config.FEED_MAX_SUBSCRIPTIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"feed subscription limit ({config.FEED_MAX_SUBSCRIPTIONS}) reached",
                )
            cur = await db.execute(
                "INSERT OR IGNORE INTO user_subscriptions (user_id, kind, value, created_at)"
                " VALUES (?, ?, ?, ?)",
                (user_id, kind, value, ts),
            )
        return cur.rowcount > 0

    async def remove_subscription(self, user_id: str, kind: str, value: str) -> bool:
        async with self._unit_of_work() as db:
            cur = await db.execute(
                "DELETE FROM user_subscriptions WHERE user_id = ? AND kind = ? AND value = ?",
                (user_id, kind, value),
            )
        return cur.rowcount > 0


def _require_store() -> FeedStore:
    if store is None:
        raise HTTPException(status_code=503, detail="feed store not initialized")
    return store


@router.get("/subscriptions", response_model=SubscriptionsResponse)
async def list_subscriptions(request: Request) -> SubscriptionsResponse:
    """All of the caller's subscriptions, oldest first."""
    user_id = request.state.user_id
    subs = await _require_store().list_subscriptions(user_id)
    return SubscriptionsResponse(subscriptions=subs)


@router.post("/subscriptions", response_model=AddResponse)
async def add_subscription(body: SubscriptionIn, request: Request) -> AddResponse:
    """Add a subscription. Idempotent: re-adding returns added=False, and adding
    past the cap is a 400 only for a genuinely new (user_id, kind, value) row."""
    user_id = request.state.user_id
    added = await _require_store().add_subscription(user_id, body.kind, body.value)
    return AddResponse(ok=True, added=added)


@router.delete("/subscriptions", response_model=RemoveResponse)
async def remove_subscription(body: SubscriptionIn, request: Request) -> RemoveResponse:
    """Remove a subscription. Idempotent: a missing (kind, value) returns
    removed=False rather than an error."""
    user_id = request.state.user_id
    removed = await _require_store().remove_subscription(user_id, body.kind, body.value)
    return RemoveResponse(ok=True, removed=removed)


def _build_feed_filter(subs: Sequence[SubscriptionOut]) -> Filter:
    should = [
        FieldCondition(key=KIND_FACET_KEY[kind], match=MatchAny(any=values))
        for kind, values in _group_by_kind(subs).items()
    ]
    return Filter(should=should)


async def _scroll_feed(subs: Sequence[SubscriptionOut], limit: int) -> list[FeedSummary]:
    """One Qdrant scroll ordered by published_date desc, filtered by ANY sub."""
    from app import main  # lazy, avoids a module-level import app.main

    points, _ = await main.state["qdrant"].scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=_build_feed_filter(subs),
        limit=limit,
        offset=None,
        order_by={"key": "published_date", "direction": "desc"},
        with_payload=main._PAYLOAD_FIELDS,
        with_vectors=False,
    )
    return [_point_to_summary(p) for p in points]


@router.get("", response_model=FeedResponse)
async def get_feed(
    request: Request,
    limit: int = Query(default=config.FEED_DEFAULT_LIMIT, ge=1, le=50),
) -> FeedResponse:
    """Most recent articles matching ANY of the caller's subscriptions.

    The response has the same shape as a search response (``results`` +
    ``note``). With no subscriptions or no matches it returns an honest EMPTY
    ``results`` list and an explanatory note -- never fabricated articles.

    Scroll with ``order_by`` on ``published_date`` so the payload index returns
    the newest ``limit`` directly instead of materialising the full corpus.
    """
    user_id = request.state.user_id
    subs = await _require_store().list_subscriptions(user_id)
    if not subs:
        return FeedResponse(results=[], note=EMPTY_SUBSCRIPTIONS_NOTE)
    results = await _scroll_feed(subs, limit)
    if not results:
        return FeedResponse(results=[], note=EMPTY_RESULTS_NOTE)
    return FeedResponse(results=results, note=None)
