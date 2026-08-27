"""
Helpers shared by the index scripts (fetch/build/update) and the API.

* compose_dense_text(): input for the DENSE embedder. Title + facets
  (authors, industry, dealtype) + summary — metadata first so the facet values
  survive the embedder's 512-token truncation. NO body: the dense transformer
  is token-bound, so keeping this short keeps builds fast, and semantic
  matching keys on the headline + facets + summary.
* compose_sparse_text(): input for the SPARSE (BM25/lexical) embedder. Metadata
  + summary + FULL body, so keyword matches inside the article body stay
  searchable at cheap lexical cost.
* split_names(): normalizes the comma/delimiter-separated *_names columns into
  a clean, de-duplicated list (used for the payload facet fields).
* normalize_date(): converts MySQL datetime values into RFC 3339 so Qdrant's
  DATETIME payload index, range filters and recency blending all parse them.
* record_from_row(): builds the canonical indexed record from a MySQL row, so
  fetch_data.py and update_index.py build identical payloads.
"""
import html
import json
import re
from datetime import UTC, datetime

from app.config import config

# vcc_frontend schema -> canonical record mapping. The table/pk come from config.
# external_url wins over canonical_url for the canonical article link.
EXTERNAL_URL_SQL = "COALESCE(NULLIF(external_url, ''), NULLIF(canonical_url, ''))"


def split_names(value) -> list[str]:
    """Split a *_names column ('TMT,Technology' or JSON-like list) into values."""
    if not value:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, TypeError):
            pass
    seen: list[str] = []
    for part in re.split(r"[,|/]+", s):
        p = part.strip()
        if p and p not in seen:
            seen.append(p)
    return seen


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
        "author_names": split_names(row["author_names"]),
        "industry_names": split_names(row["industry_names"]),
        "dealtype_names": split_names(row["dealtype_names"]),
    }


def normalize_date(value):
    """MySQL datetime -> RFC 3339 string (or None), safe for Qdrant/parsing.

    The MySQL `publish` column is a naive wall-clock value in the database's
    local timezone (no offset is stored). We deliberately do NOT attach a
    hardcoded UTC offset here: assuming UTC would shift every date/year filter
    by the DB's tz offset (e.g. +5:30 for IST) versus what users expect.

    Naive datetimes are kept naive. Qdrant interprets a tz-less RFC 3339
    timestamp as UTC, which matches how the filter side (`_parse_date` in
    main.py) treats naive user input, so stored values and range filters
    compare on a consistent wall-clock basis. If a configured timezone ever
    becomes available, attach that tz here instead of assuming UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        # Normalize any tz-aware input to UTC (keeping the +00:00 suffix) so the
        # stored string is consistent for the tiebreaker in main.py, which strips
        # the +00:00/Z suffix. Naive inputs (local wall-clock from MySQL) are left
        # untouched — attaching a UTC offset here would shift every date filter.
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC)
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T", 1) if "T" not in s else s)
        except ValueError:
            return s
    return dt.isoformat()


def _seg(p: str) -> str:
    return p.strip().rstrip(".")


def _join_vals(vals):
    if not vals:
        return ""
    return ", ".join(v for v in vals if v)


def _lead(rec: dict) -> str:
    """Title + facet values (authors, industry, dealtype), metadata first."""
    return ". ".join(
        _seg(p)
        for p in [
            rec.get("title"),
            _join_vals(rec.get("author_names")),
            _join_vals(rec.get("industry_names")),
            _join_vals(rec.get("dealtype_names")),
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