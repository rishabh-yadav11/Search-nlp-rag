"""Shared helpers for the ops/ETL scripts.

One definition of the UTC ``log()``, the redacting URL printer, the Qdrant
point builder, the payload index set and the MySQL pool, so two scripts cannot
write a *different* payload for the same article -- divergence that is silent
and expensive to detect afterwards. Resolved by callers through the
``sys.path`` bootstrap below, so ``python scripts/<name>.py`` keeps working
from any working directory.
"""
import os
import sys
from datetime import UTC, datetime
from urllib.parse import urlparse

# Mirrors the bootstrap every script in this directory performs, so this module
# is importable on its own (``app`` lives one directory up).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import PayloadSchemaType, PointStruct, SparseVector

from app.config import config


def log(msg: str):
    """Print one UTC-timestamped progress line (the ops scripts' shared format)."""
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def redact_url(url: str) -> str:
    """Return the URL with userinfo and query-param secrets stripped.

    Ops output is pasted into tickets and chat, and a service URL can carry a
    password in its userinfo or a token in its query string, so no script may
    print one raw. Any parse failure returns a placeholder rather than the
    original, so a malformed, possibly secret-bearing value cannot leak.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return parsed._replace(netloc=netloc, query="").geturl()
    except ValueError:
        return "<redacted>"


def make_point(rec: dict, dvec, svec) -> PointStruct:
    """Build the Qdrant point for one indexed record.

    ``rec`` is the canonical record from ``app.index_text.record_from_row``;
    ``dvec`` and ``svec`` are its dense and sparse embeddings. The key set is
    fixed here, so a field not stored by this function is stored by no script.

    Values on a KEYWORD-indexed field are normalised to ``str``, never ``None``:
    a field that is ``str`` on some points and ``None`` on others is indexed
    inconsistently. The read side maps a falsy value back to ``None``.
    """
    payload = {
        "title": rec["title"],
        "url": rec["url"],
        "published_date": rec.get("published_date"),
        "category": rec.get("category"),
        "content_type": rec.get("content_type") or "",
        "summary": rec.get("summary") or "",
        "body": (rec.get("body") or "")[: config.BODY_CHAR_LIMIT],
        "author_names": rec.get("author_names") or [],
        "industry_names": rec.get("industry_names") or [],
        "dealtype_names": rec.get("dealtype_names") or [],
        "tag_names": rec.get("tag_names") or [],
    }
    return PointStruct(
        id=rec["id"],
        vector={
            "dense": dvec.tolist(),
            "sparse": SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
        },
        payload=payload,
    )


def create_payload_indexes(client):
    """Create the payload index for every field the search read path filters on.

    A stored field is not filterable until its payload index exists, so this
    list must cover every field ``main.build_facet_filter`` builds a condition
    on. It lives beside ``make_point`` on purpose: the key set and the index set
    are two halves of one contract. Re-creating an existing index is a no-op in
    Qdrant, so ``build_index.py`` calls this on a resumed build too -- otherwise
    a field added after the collection was built stays unfilterable.
    """
    for field, schema in (
        ("category", PayloadSchemaType.KEYWORD),
        ("published_date", PayloadSchemaType.DATETIME),
        ("author_names", PayloadSchemaType.KEYWORD),
        ("industry_names", PayloadSchemaType.KEYWORD),
        ("dealtype_names", PayloadSchemaType.KEYWORD),
        ("tag_names", PayloadSchemaType.KEYWORD),
        # content_type is single-valued, so it gets a plain KEYWORD index (the
        # *_names fields are keyword too, but they hold lists).
        ("content_type", PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(config.QDRANT_COLLECTION, field, schema)


async def make_pool(*, maxsize: int = 3, connect_timeout: int | None = None):
    """Open an autocommit MySQL pool from the app's configured credentials.

    ``maxsize`` and ``connect_timeout`` are passed through because call sites
    legitimately differ. ``read_timeout``/``write_timeout`` are deliberately
    absent: the installed aiomysql rejects them with a TypeError before any
    connection is attempted, and exposes no equivalent.
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
