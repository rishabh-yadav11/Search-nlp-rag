import re
from datetime import date

_CURRENT_YEAR = date.today().year

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_YEAR_SPAN_RE = re.compile(r"\b(20\d{2}|19\d{2})\s*(?:-|to|through)\s*(20\d{2}|19\d{2})\b", re.IGNORECASE)
_LAST_YEAR_RE = re.compile(r"\b(?:the\s+)?last\s+year\b|\bprevious\s+year\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)
_FLASHBACK_RE = re.compile(r"\bflashback\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
_TOP_HINT_RE = re.compile(r"\btop\s+(\d{1,2})\b|\b(best|leading|biggest|largest|top)\b", re.IGNORECASE)
_DEFAULT_LIST_K = 10


def extract_year_range(query: str) -> tuple[str, str] | None:
    """Return (from_date, to_date) ISO strings for a year mentioned in the query
    (explicit span, explicit year, or 'last year'/'this year'), else None."""
    q = query.lower()
    m = _YEAR_SPAN_RE.search(q)
    if m:
        return (f"{m.group(1)}-01-01", f"{m.group(2)}-12-31")
    if _LAST_YEAR_RE.search(q):
        y = _CURRENT_YEAR - 1
        return (f"{y}-01-01", f"{y}-12-31")
    if _THIS_YEAR_RE.search(q):
        y = _CURRENT_YEAR
        return (f"{y}-01-01", f"{y}-12-31")
    m = _YEAR_RE.search(q)
    if m:
        y = int(m.group(1))
        return (f"{y}-01-01", f"{y}-12-31")
    return None


def top_k_hint(query: str) -> int | None:
    """Suggested top_k from a 'top N' in the query, or a small default for a
    generic top/best intent without a number. None when no list intent."""
    m = re.search(r"\btop\s+(\d{1,2})\b", query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if _TOP_HINT_RE.search(query):
        return _DEFAULT_LIST_K
    return None


def rewrite_year_in_review(query: str) -> tuple[str, bool]:
    """For 'top/best <topic> in <year>' style queries, rewrite to surface the
    year-in-review ('Flashback <year>') articles. Returns (query, changed)."""
    topic = extract_list_topic(query)
    yr = _year_of(query)
    if yr is None or topic is None:
        return query, False
    new_q = f"Flashback {yr} {topic}".strip()
    return new_q, new_q != query


def _year_of(query: str) -> int | None:
    """The year referenced by the query (explicit, span, last/this year, or an
    explicit 'Flashback <year>' prefix), else None."""
    m = _FLASHBACK_RE.search(query)
    if m:
        return int(m.group(1))
    rng = extract_year_range(query)
    if rng is not None:
        return int(rng[0][:4])
    return None


def extract_list_topic(query: str) -> str | None:
    """The bare topic of a top-N query with year/time words removed, e.g.
    'top 3 unicorns created in 2025' -> 'unicorns created'. None when the
    query is not a top/best/list intent."""
    if _TOP_HINT_RE.search(query) is None:
        return None
    stripped = _YEAR_SPAN_RE.sub(" ", query)
    stripped = _YEAR_RE.sub(" ", stripped)
    stripped = _LAST_YEAR_RE.sub(" ", stripped)
    stripped = _THIS_YEAR_RE.sub(" ", stripped)
    stripped = _TOP_HINT_RE.sub(" ", stripped)
    stripped = _FLASHBACK_RE.sub(" ", stripped)
    stripped = re.sub(
        r"\bof\b|\bin\b|\bfor\b|\bto\b|\byear\b|\byears\b|\bflashback\b", " ", stripped, flags=re.IGNORECASE
    )
    stripped = re.sub(r"[\s-]+", " ", stripped).strip()
    topic = stripped.strip()
    return topic or None
