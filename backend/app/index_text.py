"""
Helpers shared by the index scripts (fetch/build/update) and the API.

* compose_index_text(): builds the searchable text for an article. Metadata
  (title, authors, industry, dealtype) comes FIRST, then summary, then body,
  so the embedder's 512-token truncation never drops the facet values that
  candidates' queries will match on.
* split_names(): normalizes the comma/delimiter-separated *_names columns into
  a clean, de-duplicated list (used for the payload facet fields).
* normalize_date(): converts MySQL datetime values into RFC 3339 so Qdrant's
  DATETIME payload index, range filters and recency blending all parse them.
"""
import re
from datetime import datetime, timezone

from app.config import config


def split_names(value) -> list[str]:
    """Split a *_names column ('TMT,Technology' or JSON-like list) into values."""
    if not value:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = __import__("json").loads(s)
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


def normalize_date(value):
    """MySQL datetime -> RFC 3339 string (or None), safe for Qdrant/parsing."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T", 1) if "T" not in s else s)
        except ValueError:
            return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def compose_index_text(rec: dict) -> str:
    """Searchable text for an article: metadata before summary before body."""
    def seg(p: str) -> str:
        return p.strip().rstrip(".")

    def join_vals(vals):
        return ", ".join(v for v in vals if v)

    lead = ". ".join(
        seg(p)
        for p in [
            rec.get("title"),
            join_vals(rec.get("author_names")),
            join_vals(rec.get("industry_names")),
            join_vals(rec.get("dealtype_names")),
        ]
        if p
    )
    rest = ". ".join(
        seg(p)
        for p in [
            (rec.get("summary") or "").strip(),
            (rec.get("body") or "").strip(),
        ]
        if p
    )
    if not lead and not rest:
        return ""
    text = (lead + ((". " + rest) if rest else "")).strip()
    return text[: config.EMBED_CHAR_LIMIT]