"""
Helpers shared by the index scripts (fetch/build/update) and the API.

* compose_dense_text(): DENSE embedder input. Title + facets (authors,
  industry, dealtype, tags) + summary -- metadata first so facet values
  survive the embedder's 512-token truncation. NO body: the dense transformer
  is token-bound, and matching keys on headline + facets + summary.
* compose_sparse_text(): SPARSE (BM25/lexical) embedder input. The same
  metadata lead (so a tag is a lexical term too) + summary + FULL body, so
  in-body keyword matches stay searchable at cheap lexical cost.
* split_names(): normalizes the delimiter-separated *_names columns into a
  de-duplicated list (used for the payload facet fields).
* normalize_date(): MySQL datetime -> RFC 3339, which Qdrant's DATETIME index,
  range filters and recency blending all parse.
* record_from_row(): builds the canonical indexed record from a MySQL row, so
  the index scripts build identical payloads.
"""
import html
import json
import logging
import re
from datetime import UTC, datetime

from app.config import config

logger = logging.getLogger(__name__)

# vcc_frontend schema -> canonical record mapping. The table/pk come from config.
# external_url wins over canonical_url for the canonical article link.
EXTERNAL_URL_SQL = "COALESCE(NULLIF(external_url, ''), NULLIF(canonical_url, ''))"


def split_names(value) -> list[str]:
    """Split a *_names column ('TMT,Technology' or JSON-like list) into values.

    Always returns a flat, deduplicated list of non-empty strings (never None).
    """
    if not value:
        return []
    if isinstance(value, list):
        items = _flatten_names(value)
    else:
        s = str(value).strip()
        if not s:
            return []
        items = []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = _flatten_names(parsed)
            except (ValueError, TypeError):
                items = []
        if not items:
            for part in re.split(r"[,|]+", s):
                items.extend(_flatten_names([part]))
    flat: list[str] = []
    for p in items:
        if p and p not in flat:
            flat.append(p)
    return flat


def _flatten_names(values) -> list[str]:
    """Recursively flatten arbitrary nested lists into a clean str list."""
    out: list[str] = []
    for x in values:
        if isinstance(x, list):
            out.extend(_flatten_names(x))
        else:
            out.append(str(x).strip())
    return out


def clean(text) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace."""
    if text is None:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(text))
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def record_from_row(row: dict) -> dict:
    """Build the canonical indexed record dict from a MySQL row (DictCursor).

    Fields are shared verbatim by fetch_data.py (dump to articles.jsonl),
    update_index.py (incremental upsert) and backfill_summary.py (payload repair).
    """
    return {
        "id": row["feid"],
        "title": clean(row["title"]),
        "summary": clean(row["summary"]),
        "body": clean(row["body"])[: config.BODY_CHAR_LIMIT],
        "url": row["ext_url"] or f"https://www.vccircle.com/{row['slug'] or row['feid']}",
        "published_date": normalize_date(row["publish"]),
        "category": (row["dealtype_names"] or row["content_type"] or "").strip(),
        "content_type": (row["content_type"] or "").strip(),
        "author_names": split_names(row["author_names"]),
        "industry_names": split_names(row["industry_names"]),
        "dealtype_names": split_names(row["dealtype_names"]),
        # row.get(), not row[...]: backfill_body.py and backfill_missing.py reuse
        # this builder with a narrower SELECT lacking tag_names.
        "tag_names": split_names(row.get("tag_names")),
    }


def normalize_date(value):
    """MySQL datetime -> RFC 3339 string (or None), safe for Qdrant/parsing.

    The MySQL `publish` column is a naive wall-clock value in the database's
    local timezone. No offset is attached on purpose: assuming UTC would shift
    every date/year filter by the DB's offset. Qdrant reads a tz-less RFC 3339
    timestamp as UTC, which matches how the filter side (``_parse_date`` in
    main.py) treats naive user input, so both compare on the same wall clock.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        # Tz-aware input goes to UTC, keeping the +00:00 suffix main.py's
        # tiebreaker strips. Naive input is left alone -- attaching an offset
        # would shift every date filter.
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC)
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T", 1) if "T" not in s else s)
        except ValueError:
            logger.warning("normalize_date: unparseable date value %r; skipping", value)
            return None
    return dt.isoformat()


def _seg(p: str) -> str:
    return p.strip().rstrip(".")


def _join_vals(vals):
    if not vals:
        return ""
    return ", ".join(v for v in vals if v)


def _lead(rec: dict) -> str:
    """Title + facet values (authors, industry, dealtype, tags), metadata first."""
    return ". ".join(
        _seg(p)
        for p in [
            rec.get("title"),
            _join_vals(rec.get("author_names")),
            _join_vals(rec.get("industry_names")),
            _join_vals(rec.get("dealtype_names")),
            _join_vals(rec.get("tag_names")),
        ]
        if p
    )


def compose_dense_text(rec: dict) -> str:
    """Dense embedder input: title + facets + summary (no body)."""
    lead = _lead(rec)
    summary = _seg((rec.get("summary") or "").strip())
    text = (lead + ((". " + summary) if summary else "")).strip()
    return text[: config.EMBED_DENSE_CHAR_LIMIT]


def compose_sparse_text(rec: dict) -> str:
    """Sparse (BM25/lexical) embedder input: metadata + summary + full body."""
    lead = _lead(rec)
    rest = ". ".join(
        _seg(p)
        for p in [
            (rec.get("summary") or "").strip(),
            (rec.get("body") or "").strip(),
        ]
        if p
    )
    text = (lead + ((". " + rest) if rest else "")).strip()
    return text[: config.EMBED_CHAR_LIMIT]