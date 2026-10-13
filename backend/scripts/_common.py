
import os
import sys
from datetime import UTC, datetime
from urllib.parse import urlparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import PayloadSchemaType, PointStruct, SparseVector

from app.config import config


def log(msg: str):

    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def redact_url(url: str) -> str:

    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return parsed._replace(netloc=netloc, query="").geturl()
    except ValueError:
        return "<redacted>"


def make_point(rec: dict, dvec, svec) -> PointStruct:

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

    for field, schema in (
        ("category", PayloadSchemaType.KEYWORD),
        ("published_date", PayloadSchemaType.DATETIME),
        ("author_names", PayloadSchemaType.KEYWORD),
        ("industry_names", PayloadSchemaType.KEYWORD),
        ("dealtype_names", PayloadSchemaType.KEYWORD),
        ("tag_names", PayloadSchemaType.KEYWORD),


        ("content_type", PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(config.QDRANT_COLLECTION, field, schema)


async def make_pool(*, maxsize: int = 3, connect_timeout: int | None = None):

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
