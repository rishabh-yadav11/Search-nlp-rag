"""Shared helpers for the ops/ETL scripts.

Every script in this directory used to carry its own copy of the same three
helpers: a UTC ``log()``, a Qdrant ``PointStruct`` builder, and an
``aiomysql.create_pool`` call. Six copies of the first and three of the second
meant an ops run of one script could write a *different* payload for the same
article than another script — silent index divergence, expensive to detect
after the fact. This module is the single definition of all three.

Follows the same convention as ``qdrant_backup.py``: a plain importable helper
module with no ``__main__`` block, resolved by the caller through the
``sys.path`` bootstrap below (so ``python scripts/<name>.py`` keeps working
from any working directory).

Used by: build_index.py, update_index.py, backfill_body.py,
backfill_missing.py, backfill_summary.py, fetch_data.py, build_query_vocab.py,
qdrant_backup.py.
"""
import os
import sys
from datetime import UTC, datetime

# Mirrors the bootstrap every script in this directory already performs, so this
# module is importable on its own (``app`` lives one directory up).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import PointStruct, SparseVector

from app.config import config


def log(msg: str):
    """Print one UTC-timestamped progress line (the ops scripts' shared format)."""
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


# The payload keys every index write path stores. Kept as an explicit tuple so
# the regression test can assert the exact set rather than a subset.
PAYLOAD_KEYS = (
    "title",
    "url",
    "published_date",
    "category",
    "summary",
    "body",
    "author_names",
    "industry_names",
    "dealtype_names",
)


def make_point(rec: dict, dvec, svec, *, content_type: str | None = None) -> PointStruct:
    """Build the Qdrant point for one indexed record.

    ``rec`` is the canonical record produced by ``app.index_text.record_from_row``
    (or the same dict as read back from ``data/articles.jsonl``); ``dvec`` and
    ``svec`` are the dense and sparse embeddings for it.

    ``content_type`` is an explicit opt-in: it is written to the payload only
    when a caller passes it. The record dict carries the value (it comes from
    the same ``record_from_row`` source as every other field), but no write path
    has ever persisted it, and the read side in ``app/main.py`` reads it from the
    payload. Callers therefore have to ask for the key by name — a payload can
    no longer lose a field by accident. See the ``content_type`` payload
    tracking issue for the end-to-end fix.
    """
    payload = {
        "title": rec["title"],
        "url": rec["url"],
        "published_date": rec.get("published_date"),
        "category": rec.get("category"),
        "summary": rec.get("summary") or "",
        "body": (rec.get("body") or "")[: config.BODY_CHAR_LIMIT],
        "author_names": rec.get("author_names") or [],
        "industry_names": rec.get("industry_names") or [],
        "dealtype_names": rec.get("dealtype_names") or [],
    }
    if content_type is not None:
        payload["content_type"] = content_type
    return PointStruct(
        id=rec["id"],
        vector={
            "dense": dvec.tolist(),
            "sparse": SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
        },
        payload=payload,
    )


async def make_pool(*, maxsize: int = 3, connect_timeout: int | None = None):
    """Open an autocommit MySQL pool from the app's configured credentials.

    Every script that talks to MySQL reads the same ``config.MYSQL_*`` settings
    (MYSQL_HOST/PORT/USER/PASSWORD/DATABASE), so the connection parameters are
    taken from one place here rather than being restated per script.

    ``maxsize`` (pool ceiling; ``minsize`` is always 1) and ``connect_timeout``
    are passed through because the call sites legitimately differ. Note that
    ``read_timeout``/``write_timeout`` are deliberately absent: the installed
    aiomysql does not accept them, and any call passing them raises
    ``TypeError: connect() got an unexpected keyword argument`` before a
    connection is ever attempted.
    """
    import aiomysql

    return await aiomysql.create_pool(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE,
        autocommit=True,
        minsize=1,
        maxsize=maxsize,
        connect_timeout=connect_timeout,
    )
