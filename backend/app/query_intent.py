import calendar
import re
from datetime import date

_CURRENT_YEAR = date.today().year

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_YEAR_SPAN_RE = re.compile(r"\b(20\d{2}|19\d{2})\s*(?:-|to|through)\s*(20\d{2}|19\d{2})\b", re.IGNORECASE)
_LAST_YEAR_RE = re.compile(r"\b(?:the\s+)?last\s+year\b|\bprevious\s+year\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)
_FLASHBACK_RE = re.compile(r"\bflashback\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
_TOP_N_RE = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
_TOP_HINT_RE = re.compile(r"\b(best|leading|biggest|largest|top)\b", re.IGNORECASE)
_DEFAULT_LIST_K = 10
# Filler/time words dropped when extracting a bare topic from a query
_NOISE_WORDS_RE = re.compile(
    r"\bof\b|\bin\b|\bfor\b|\bto\b|\bmonth\b|\bmonths\b|\byear\b|\byears\b|\bflashback\b",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
# month(+optional year) with optional filler words: "january 2025", "of month january 2025", "in jan"
_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\b[^.\d]*(?:\b(20\d{2}|19\d{2})\b)?",
    re.IGNORECASE,
)


def extract_month_range(query: str) -> tuple[str, str] | None:
    """Return (from_date, to_date) ISO strings for a specific month in the query,
    e.g. 'january 2025' -> ('2025-01-01', '2025-01-31'). Year defaults to the
    current year when not given. None when no month is mentioned."""
    q = query.lower()
    m = _MONTH_RE.search(q)
    if not m:
        return None
    month = _MONTHS[m.group(1)]
    ym = _MONTH_YEAR_RE.search(q)
    year = int(ym.group(2)) if ym and ym.group(2) else _CURRENT_YEAR
    last_day = calendar.monthrange(year, month)[1]
    return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}")


def extract_year_range(query: str) -> tuple[str, str] | None:
    """Return (from_date, to_date) ISO strings for a time window mentioned in the
    query: a specific month ('january 2025' -> month range), a year span, an
    explicit year, or 'last year'/'this year'. Month takes precedence."""
    q = query.lower()
    month_range = extract_month_range(q)
    if month_range is not None:
        return month_range
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


def suggested_top_k(query: str) -> int | None:
    """Suggested top_k from a 'top N' in the query, or a small default for a
    generic top/best intent without a number. None when no list intent."""
    m = _TOP_N_RE.search(query)
    if m:
        return int(m.group(1))
    if _TOP_HINT_RE.search(query):
        return _DEFAULT_LIST_K
    return None


def rewrite_year_in_review(query: str) -> tuple[str, bool]:
    """For 'top/best <topic> in <year>' style queries, rewrite to surface the
    year-in-review ('Flashback <year>') articles. Returns (query, changed).

    Queries that mention a specific MONTH are NOT rewritten: Flashback articles
    are annual roundups, so a month-scoped query should match the month's actual
    articles instead."""
    if extract_month_range(query) is not None:
        return query, False
    topic = extract_list_topic(query)
    yr = _referenced_year(query)
    if yr is None or topic is None:
        return query, False
    new_q = f"Flashback {yr} {topic}".strip()
    return new_q, new_q != query


def _referenced_year(query: str) -> int | None:
    """The year referenced by the query (explicit, span, last/this year, or an
    explicit 'Flashback <year>' prefix), else None."""
    m = _FLASHBACK_RE.search(query)
    if m:
        return int(m.group(1))
    rng = extract_year_range(query)
    if rng is not None:
        return int(rng[0][:4])
    return None


def month_query_topic(query: str) -> str | None:
    """Cleaned retrieval/rerank query for a month-scoped query, e.g.
    'top pharma deals of month january 2025' -> 'pharma deals'. The date filter
    already scopes the month, so dropping the 'top/of/month/year' words lets the
    embeddings and cross-encoder focus on the actual topic. None when the query
    doesn't mention a specific month."""
    if extract_month_range(query) is None:
        return None
    topic = extract_list_topic(query)
    if topic:
        return topic
    return _strip_noise_words(query)


def extract_list_topic(query: str) -> str | None:
    """The bare topic of a top-N query with year/time words removed, e.g.
    'top 3 unicorns created in 2025' -> 'unicorns created'. None when the
    query is not a top/best/list intent."""
    if _TOP_HINT_RE.search(query) is None and _TOP_N_RE.search(query) is None:
        return None
    stripped = _YEAR_SPAN_RE.sub(" ", query)
    stripped = _YEAR_RE.sub(" ", stripped)
    stripped = _LAST_YEAR_RE.sub(" ", stripped)
    stripped = _THIS_YEAR_RE.sub(" ", stripped)
    stripped = _TOP_N_RE.sub(" ", stripped)
    stripped = _TOP_HINT_RE.sub(" ", stripped)
    stripped = _FLASHBACK_RE.sub(" ", stripped)
    stripped = _MONTH_RE.sub(" ", stripped)
    stripped = _NOISE_WORDS_RE.sub(" ", stripped)
    topic = re.sub(r"[\s-]+", " ", stripped).strip()
    return topic or None


def _strip_noise_words(query: str) -> str | None:
    """Remove month/year/time filler words, leaving the bare query text."""
    q = _MONTH_RE.sub(" ", query)
    q = _YEAR_SPAN_RE.sub(" ", q)
    q = _YEAR_RE.sub(" ", q)
    q = _NOISE_WORDS_RE.sub(" ", q)
    q = re.sub(r"[\s-]+", " ", q).strip()
    return q or None
