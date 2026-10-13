
import html
import json
import logging
import re
from datetime import UTC, datetime

from app.config import config

logger = logging.getLogger(__name__)



EXTERNAL_URL_SQL = "COALESCE(NULLIF(external_url, ''), NULLIF(canonical_url, ''))"


def split_names(value) -> list[str]:

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

    out: list[str] = []
    for x in values:
        if isinstance(x, list):
            out.extend(_flatten_names(x))
        else:
            out.append(str(x).strip())
    return out


def clean(text) -> str:

    if text is None:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(text))
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def record_from_row(row: dict) -> dict:

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


        "tag_names": split_names(row.get("tag_names")),
    }


def normalize_date(value):

    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value



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

    lead = _lead(rec)
    summary = _seg((rec.get("summary") or "").strip())
    text = (lead + ((". " + summary) if summary else "")).strip()
    return text[: config.EMBED_DENSE_CHAR_LIMIT]


def compose_sparse_text(rec: dict) -> str:

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