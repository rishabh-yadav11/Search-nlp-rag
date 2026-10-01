import calendar
import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

# 'Today' follows the Indian calendar, not the host's: between 00:00 and 05:30 IST the UTC date is
# still the previous day, which would resolve the wrong year around New Year on a UTC server.
_IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    return datetime.now(UTC)


def _today() -> date:
    return _now().astimezone(_IST).date()


def _current_year() -> int:
    return _today().year

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_YEAR_SPAN_RE = re.compile(
    r"\b(20\d{2}|19\d{2})\s*(?:-|to|through|and)\s*(20\d{2}|19\d{2})\b", re.IGNORECASE
)
# Short year span: '2024-25' -> 2024 and 2025; the trailing \b keeps it out of a 4-digit year.
_YEAR_SPAN_SHORT_RE = re.compile(
    r"\b(20\d{2}|19\d{2})\s*(?:-|to|through)\s*(\d{2})\b", re.IGNORECASE
)
_LAST_YEAR_RE = re.compile(r"\b(?:the\s+)?last\s+year\b|\bprevious\s+year\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)
_FLASHBACK_RE = re.compile(r"\bflashback\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
# Word-form counts, built longest-first so 'fourteen' wins over 'four'.
_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS = dict(_UNITS, **_TEENS, **_TENS)
for _tens_word, tens_val in _TENS.items():
    _NUMBER_WORDS[_tens_word] = tens_val
    for _unit_word, unit_val in _UNITS.items():
        _NUMBER_WORDS[f"{_tens_word}{_unit_word}"] = tens_val + unit_val
_NUMBER_WORDS["hundred"] = 100
_NUM_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_WORD_SEP = r"(?:\s+|-|\s+-\s+)"
_TOP_HINT_ALT = r"top|best|leading|biggest|largest"
_TOP_N_RE = re.compile(
    rf"\b({_TOP_HINT_ALT})\s+((?:\d{{1,4}})|(?:(?:{_NUM_WORD_ALT})(?:{_WORD_SEP}(?:{_NUM_WORD_ALT}))*))\b",
    re.IGNORECASE,
)
_TOP_HINT_RE = re.compile(r"\b(best|leading|biggest|largest|top)\b", re.IGNORECASE)
_DEFAULT_LIST_K = 10
# Superlatives that mean a ranked answer over many items. "most" counts only before an aggregation
# noun ("most of the time" is not one), and "least" is excluded after "at " so the threshold phrase
# "at least" does not read as a superlative.
_SUPERLATIVE_ALT = r"biggest|largest|highest|greatest|maximum|smallest|lowest|(?<!at\s)least"
_AGG_NOUN_ALT = (
    r"active|funded|funding|invested|investing|investments?|valuable|valued|"
    r"profitable|raised|successful|mentioned|cited|covered|popular|influential|"
    r"deal(s|making)?|acquisitive"
)
_SUPERLATIVE_RE = re.compile(
    rf"\b({_SUPERLATIVE_ALT})\b|\bmost\s+({_AGG_NOUN_ALT})\b", re.IGNORECASE
)
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
_MONTH_ALT = (
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_MONTH_RE = re.compile(rf"\b({_MONTH_ALT})\b", re.IGNORECASE)
# A span of months; each side may carry its own year.
_MONTH_SPAN_RE = re.compile(
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?\s*(?:-|to|through|and)\s*"
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?",
    re.IGNORECASE,
)
# month (+optional year), tolerating filler words between the two ("of month january 2025").
_MONTH_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALT})\b[^.\d]*(?:\b(20\d{{2}}|19\d{{2}})\b)?",
    re.IGNORECASE,
)

_QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
_QUARTER_WORD_ORDER = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
}
_Q_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
_Q_YEAR_RE = re.compile(
    r"\bq([1-4])\s*(?:of\s*)?(?:-|/|')?\s*((?:19|20)\d{2}|\d{2})\b", re.IGNORECASE
)
_QUARTER_WORD_RE = re.compile(
    r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+quarter\b(?:\s+of\s+((?:19|20)\d{2}))?",
    re.IGNORECASE,
)

# Fiscal-year references, in three named-group tiers (span / bare 'fy N' / 'fiscal year N' or
# 'fiscal N'); the grammar is shared with `_strip_time_tokens` so resolving a range and stripping
# its token cannot drift apart. An Indian FY ending in year N spans Apr (N-1) to Mar N.
_FY_RE = re.compile(
    r"\bfy\s*(?P<fy_range_start>(?:19|20)?\d{2})\s*(?:-|to|through)\s*(?P<fy_range_end>(?:19|20)?\d{2})\b"
    r"|\bfy\s*'?(?P<fy_single>(?:19|20)?\d{2})\b"
    r"|\bfiscal\s+year\s*(?P<fiscal_year>(?:19|20)?\d{2})\b"
    r"|\bfiscal\s*(?P<fiscal_bare>(?:19|20)?\d{2})\b",
    re.IGNORECASE,
)

# An event-naming year ('the 2008 crisis') is a topic reference, not a publication-date filter:
# such a query wants retrospectives written later.
_EVENT_NOUNS = (
    "crisis", "crash", "bubble", "meltdown", "recession", "slowdown", "downturn",
    "pandemic", "epidemic", "outbreak", "war", "invasion", "battle", "conflict",
    "election", "referendum", "demonetisation", "demonetization", "reforms",
    "reform", "census", "olympics", "earthquake", "tsunami", "cyclone",
    "floods", "flood", "hurricane", "massacre", "riots", "protest", "scandal",
    "coup", "default", "bankruptcy", "bailout", "goldrush", "partition",
    "independence",
)
_EVENT_NOUN_ALT = "|".join(sorted(_EVENT_NOUNS, key=len, reverse=True))
_EVENT_YEAR_RE = re.compile(
    rf"\b((?:19|20)\d{{2}})\s+(?:\w+\s+){{0,2}}({_EVENT_NOUN_ALT})s?\b", re.IGNORECASE
)
_REV_EVENT_YEAR_RE = re.compile(
    rf"\b(?:\w+\s+){{0,2}}({_EVENT_NOUN_ALT})s?\s+(?:of|in)\s+((?:19|20)\d{{2}})\b", re.IGNORECASE
)

# Chart/table request filler: it names the requested OUTPUT format, not the topic, so leaving it
# in dilutes the embedding match ('make a table of top 15 deals' retrieves on 'make a table').
# Bare 'in table' is an ordinary noun phrase, so table views need a form/format/view suffix.
_CHART_VERB = r"(?:show|draw|make|create|give|build|plot|display|present|share|convert)"
_CHART_TYPE = r"(?:bar|line|pie|column|area|pictogram|pictograph)?\s*"
_CHART_NOUN = r"(?:chart|graph|plot|diagram|pictogram|pictograph)"
_CHART_TABLE_NOUN = r"(?:tables?(?! tennis\b)|tabular|tabulated)"
_CHART_LEAD_RE = re.compile(
    rf"\b(?:{_CHART_VERB})\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?"
    rf"(?:{_CHART_TYPE}(?:{_CHART_NOUN}|{_CHART_TABLE_NOUN})\b)"
    rf"(?:\s+(?:of|for|on|about|regarding)\b)?",
    re.IGNORECASE,
)
_CHART_TRAIL_RE = re.compile(
    rf"\b(?:as|in|into|using)\s+(?:a\s+|an\s+|the\s+)?"
    rf"{_CHART_TYPE}{_CHART_NOUN}\b(?:\s+(?:form|format|view)\b)?"
    rf"|\b(?:as|in|into|using)\s+(?:(?:a|an|the)\s+(?:tabular\s+)?{_CHART_TABLE_NOUN}\b"
    rf"|(?:tabular\s+)?{_CHART_TABLE_NOUN}\b\s+(?:form|format|view)\b)",
    re.IGNORECASE,
)


# Content-type modifiers name the kind of article wanted, distinct from its dealtype/industry; the
# bare modifier becomes a keyword that main.resolve_content_type promotes from the live
# `content_type` vocabulary, so an unknown corpus degrades to no filter rather than a bogus value.
_CONTENT_TYPE_ALIASES: dict[str, str] = {
    "interview": "interview",
    "interviews": "interview",
    "video": "video",
    "videos": "video",
    "article": "article",
    "articles": "article",
    "appointment": "appointment",
    "appointments": "appointment",
    "founder": "founder",
    "founders": "founder",
    "competitor": "competitor",
    "competitors": "competitor",
}


def extract_content_type(query: str) -> str | None:
    q = query.lower()
    for alias, kw in sorted(_CONTENT_TYPE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            return kw
    return None


def _is_chart_request(text: str) -> bool:
    return bool(_CHART_LEAD_RE.search(text) or _CHART_TRAIL_RE.search(text))


def _strip_chart_filler(text: str) -> str:
    s = _CHART_LEAD_RE.sub(" ", text)
    s = _CHART_TRAIL_RE.sub(" ", s)
    return s


def _year_is_event_reference(query: str, year: int) -> bool:
    for pattern, year_group in ((_EVENT_YEAR_RE, 1), (_REV_EVENT_YEAR_RE, 2)):
        for m in pattern.finditer(query):
            if int(m.group(year_group)) == year:
                return True
    return False


def _full_year(y: int, near: int) -> int:
    """Expand a 2-digit year to 4 digits in ``near``'s century, rolling a century when that lands
    more than 50 years away ('99' near 2024 -> 1999)."""
    if y >= 100:
        return y
    base = (near // 100) * 100
    full = base + y
    if full - near > 50:
        full -= 100
    elif full - near < -50:
        full += 100
    return full


def extract_month_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) ISO strings for a month or month span ('january to march 2025'); year
    defaults to the current year, None when no month is mentioned."""
    q = query.lower()
    span = _extract_month_span(q)
    if span is not None:
        return span
    m = _MONTH_RE.search(q)
    if not m:
        return None
    month = _MONTHS[m.group(1)]
    ym = _MONTH_YEAR_RE.search(q)
    year = int(ym.group(2)) if ym and ym.group(2) else _current_year()
    last_day = calendar.monthrange(year, month)[1]
    return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}")


def _extract_month_span(query: str) -> tuple[str, str] | None:
    """A span of months as (from_date, to_date), or None. A lone written year anchors the month it
    sits beside ('dec to jan 2024' -> 2023-12..2024-01)."""
    m = _MONTH_SPAN_RE.search(query)
    if not m:
        return None
    m1, y1, y2 = _MONTHS[m.group(1)], m.group(2), m.group(4)
    m2 = _MONTHS[m.group(3)]
    crosses_year = m1 > m2
    if y1 and y2:
        start_year, end_year = int(y1), int(y2)
    elif y2:
        end_year = int(y2)
        start_year = end_year - 1 if crosses_year else end_year
    elif y1:
        start_year = int(y1)
        end_year = start_year + 1 if crosses_year else start_year
    else:
        year = _current_year()
        start_year = end_year = year
        if m1 > m2:
            end_year = year + 1
    end_last = calendar.monthrange(end_year, m2)[1]
    return (f"{start_year}-{m1:02d}-01", f"{end_year}-{m2:02d}-{end_last:02d}")


def _quarter_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) for a quarter reference ('Q1 2025', 'first quarter of 2025'), or None;
    year defaults to the current year."""
    q = query.lower()
    m = _Q_YEAR_RE.search(q)
    if m:
        qn, year = int(m.group(1)), _full_year(int(m.group(2)), _current_year())
    else:
        m = _Q_RE.search(q)
        if m:
            qn, year = int(m.group(1)), _current_year()
        else:
            m = _QUARTER_WORD_RE.search(q)
            if not m:
                return None
            qn = _QUARTER_WORD_ORDER[m.group(1).lower()]
            year = int(m.group(2)) if m.group(2) else _current_year()
    sm, em = _QUARTER_MONTHS[qn]
    return (f"{year}-{sm:02d}-01", f"{year}-{em:02d}-{calendar.monthrange(year, em)[1]:02d}")


def _fiscal_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) for a fiscal-year reference ('FY25', 'FY 2024-25'), or None."""
    q = query.lower()
    # A tier wins wherever it appears, so all matches are scanned instead of taking the leftmost
    # ('fiscal 2025 and fy 2020-2021').
    matches = list(_FY_RE.finditer(q))
    span = next((m for m in matches if m.group("fy_range_start")), None)
    if span:
        y1 = _full_year(int(span.group("fy_range_start")), _current_year())
        y2 = _full_year(int(span.group("fy_range_end")), y1)
        # 'FY 2024-25' lists the first and last fiscal year, and an end-first spelling ('fy 2025-24')
        # still ends on the larger number, so min/max rather than let _full_year roll the century.
        start = min(y1, y2)
        end = max(y1, y2)
        # min/max guarantees start <= end, so start == end is the sole inverted-window case ('fy 25-25'
        # names one year twice): widen backwards instead of matching zero rows.
        if start == end:
            start = end - 1
        return (f"{start}-04-01", f"{end}-03-31")
    single = next((m for m in matches if m.group("fy_single")), None)
    if single:
        end = _full_year(int(single.group("fy_single")), _current_year())
        return (f"{end - 1}-04-01", f"{end}-03-31")
    fiscal = next((m for m in matches if m.group("fiscal_year") or m.group("fiscal_bare")), None)
    if fiscal:
        year = fiscal.group("fiscal_year") or fiscal.group("fiscal_bare")
        end = _full_year(int(year), _current_year())
        return (f"{end - 1}-04-01", f"{end}-03-31")
    return None


# Hard rolling windows resolve to a date range so the filter excludes old evergreen articles.
# Number-bearing forms capture (count, unit) in groups 1-2 ("past 3 days") or 3-4 ("2 weeks ago").
_RECENCY_WINDOW_RE = re.compile(
    r"\b(?:today|"
    r"this\s+week|past\s+week|last\s+week|"
    r"this\s+month|past\s+month|last\s+month|"
    r"(?:past|last)\s+(\d+)\s*(day|days|week|weeks|month|months)|"
    r"(\d+)\s*(day|days|week|weeks|month|months)\s*ago)\b",
    re.IGNORECASE,
)
# Soft recency signals weight ranking rather than filtering. 'current'/'upcoming' are excluded:
# 'current account' is a finance topic, and 'upcoming' points past the corpus.
_RECENCY_INTENT_RE = re.compile(
    r"\b(latest|recent|newest|freshest|fresh|lately|breaking|of\s+late)\b",
    re.IGNORECASE,
)

# Months are fixed at 30 days, which widens a 'past month' window by up to 2 days; a multiplier that
# disagreed with `timedelta`'s month length would silently move the window boundary.
_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


def _days_ago_iso(days: int) -> str:
    return (_today() - timedelta(days=days)).isoformat()


def _month_start_iso() -> str:
    return _today().replace(day=1).isoformat()


def _week_start_iso() -> str:
    t = _today()
    return (t - timedelta(days=t.weekday())).isoformat()


def extract_recency_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) ISO strings for a hard rolling recency window ('this week', 'past 3 days'), or None."""
    m = _RECENCY_WINDOW_RE.search(query)
    if not m:
        return None
    num = m.group(1) or m.group(3)
    unit = m.group(2) or m.group(4)
    if num and unit:
        return (_days_ago_iso(int(num) * _UNIT_DAYS[unit.lower()]), _today().isoformat())
    text = m.group(0).lower()
    if "today" in text:
        return (_days_ago_iso(0), _today().isoformat())
    # 'this week'/'this month' anchor to the calendar boundary so prior-period articles are excluded;
    # other forms ('past week', 'last month') keep their rolling window.
    if "this week" in text:
        return (_week_start_iso(), _today().isoformat())
    if "this month" in text:
        return (_month_start_iso(), _today().isoformat())
    if "week" in text:
        return (_days_ago_iso(7), _today().isoformat())
    if "month" in text:
        return (_days_ago_iso(30), _today().isoformat())
    return None


def strip_recency_window(query: str) -> str:
    """Remove rolling-window recency phrases ('this week', 'past 3 days'); only after
    ``extract_recency_range``, since stripping first discards the window the date filter is built
    from."""
    return _RECENCY_WINDOW_RE.sub(" ", query).strip()


def strip_recency_intent(query: str) -> str:
    """Remove soft recency signals ('latest', 'recent') from the retrieval text only;
    ``is_recency_intent`` reads the original query."""
    return _RECENCY_INTENT_RE.sub(" ", query).strip()


def is_recency_intent(query: str) -> bool:
    return bool(_RECENCY_INTENT_RE.search(query))


def extract_year_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) ISO strings for a time window in the query -- month, month span, fiscal
    year, quarter, year span, explicit year, 'last/this year' -- tried in that precedence order."""
    q = query.lower()
    month_range = extract_month_range(q)
    if month_range is not None:
        return month_range
    fy = _fiscal_range(q)
    if fy is not None:
        return fy
    quarter = _quarter_range(q)
    if quarter is not None:
        return quarter
    m = _YEAR_SPAN_RE.search(q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        # A descending span ('2025 to 2024') is the same window end-first; emitting it inverted
        # (from > to) would match nothing.
        start, end = min(y1, y2), max(y1, y2)
        return (f"{start}-01-01", f"{end}-12-31")
    m = _YEAR_SPAN_SHORT_RE.search(q)
    if m:
        start = int(m.group(1))
        end = _full_year(int(m.group(2)), start)
        start, end = min(start, end), max(start, end)
        return (f"{start}-01-01", f"{end}-12-31")
    if _LAST_YEAR_RE.search(q):
        y = _current_year() - 1
        return (f"{y}-01-01", f"{y}-12-31")
    if _THIS_YEAR_RE.search(q):
        y = _current_year()
        return (f"{y}-01-01", f"{y}-12-31")
    m = _YEAR_RE.search(q)
    if m:
        for ym in _YEAR_RE.finditer(q):
            year = int(ym.group(1))
            if not _year_is_event_reference(q, year):
                return (f"{year}-01-01", f"{year}-12-31")
        return None
    return None


def _top_n_to_int(phrase: str) -> int | None:
    if phrase.isdigit():
        return int(phrase)
    total = 0
    for token in re.split(r"[\s-]+", phrase.strip()):
        value = _NUMBER_WORDS.get(token)
        if value is None:
            return None
        total = (total or 1) * value if value == 100 else total + value
    return total


def normalize_word_numbers(query: str) -> str:
    """Rewrite word-form counts after a list hint to digits ('top ten ipo' -> 'top 10 ipo'): the
    literal word matches titles ('Ten Sports') where the digit form does not."""
    def _replace(m: re.Match) -> str:
        n = _top_n_to_int(m.group(2))
        return f"{m.group(1)} {n}" if n is not None else m.group(0)

    return _TOP_N_RE.sub(_replace, query)


def suggested_top_k(query: str) -> int | None:
    """Suggested top_k from a 'top N' in the query, else ``_DEFAULT_LIST_K`` for a bare list or
    superlative intent so chat fetches enough articles to rank; None when no list intent."""
    m = _TOP_N_RE.search(query)
    if m:
        n = _top_n_to_int(m.group(2))
        if n is not None:
            return n
    if _TOP_HINT_RE.search(query) or _is_superlative(query):
        return _DEFAULT_LIST_K
    return None


def _is_superlative(query: str) -> bool:
    return bool(_SUPERLATIVE_RE.search(query))


def is_aggregation_intent(query: str) -> bool:
    return suggested_top_k(query) is not None


def _strip_time_tokens(text: str) -> str:
    """Remove fiscal/quarter/month/year filler tokens, longest-first so a compound token
    ('FY 2024-25', 'jan to march') goes before its parts; chart/table filler goes first."""
    if _is_chart_request(text):
        text = _strip_chart_filler(text)
    s = _FY_RE.sub(" ", text)
    s = _QUARTER_WORD_RE.sub(" ", s)
    s = _Q_RE.sub(" ", s)
    s = re.sub(r"'(\d{2})\b", " ", s)
    s = _MONTH_SPAN_RE.sub(" ", s)
    s = _YEAR_SPAN_SHORT_RE.sub(" ", s)
    s = _YEAR_SPAN_RE.sub(" ", s)
    s = _MONTH_RE.sub(" ", s)
    s = _LAST_YEAR_RE.sub(" ", s)
    s = _THIS_YEAR_RE.sub(" ", s)
    s = _YEAR_RE.sub(" ", s)
    s = _FLASHBACK_RE.sub(" ", s)
    s = _NOISE_WORDS_RE.sub(" ", s)
    return s


def rewrite_year_in_review(query: str) -> tuple[str, bool]:
    """Rewrite 'top/best <topic> in <year>' to surface 'Flashback <year>' articles; returns (query,
    changed). Month-scoped and range/fiscal/quarter queries want their own span, not a roundup."""
    if extract_month_range(query) is not None:
        return query, False
    topic = extract_list_topic(query)
    yr = _referenced_year(query)
    if yr is None or topic is None:
        return query, False
    new_q = f"Flashback {yr} {topic}".strip()
    return new_q, new_q != query


def _referenced_year(query: str) -> int | None:
    """The single year referenced, or None when the query spans more than one (range, fiscal year,
    quarter) and must not collapse into a roundup."""
    m = _FLASHBACK_RE.search(query)
    if m:
        return int(m.group(1))
    if _YEAR_SPAN_RE.search(query) or _YEAR_SPAN_SHORT_RE.search(query):
        return None
    if _fiscal_range(query) is not None or _quarter_range(query) is not None:
        return None
    rng = extract_year_range(query)
    if rng is not None:
        start, end = int(rng[0][:4]), int(rng[1][:4])
        if start == end:
            return start
    return None


def range_query_topic(query: str) -> str | None:
    """Bare topic for a query scoped by an auto date range that is not a plain single year ('top 15
    deals in Q1 2025' -> 'deals'); the date filter already scopes the period. None for single-year
    queries (which keep their Flashback rewrite) or unfiltered queries."""
    if extract_month_range(query) is not None:
        return extract_list_topic(query) or _strip_noise_words(query)
    if _quarter_range(query) is not None or _fiscal_range(query) is not None:
        return extract_list_topic(query) or _strip_noise_words(query)
    if _YEAR_SPAN_RE.search(query) or _YEAR_SPAN_SHORT_RE.search(query):
        return extract_list_topic(query) or _strip_noise_words(query)
    return None


def extract_list_topic(query: str) -> str | None:
    if _TOP_HINT_RE.search(query) is None and _TOP_N_RE.search(query) is None:
        return None
    stripped = _strip_time_tokens(query)
    stripped = _TOP_N_RE.sub(" ", stripped)
    stripped = _TOP_HINT_RE.sub(" ", stripped)
    topic = re.sub(r"[\s-]+", " ", stripped).strip()
    return topic or None


def _strip_noise_words(query: str) -> str | None:
    q = _strip_time_tokens(query)
    q = re.sub(r"[\s-]+", " ", q).strip()
    return q or None


# Acquisition relation direction. "who acquired X?" names X as the acquired company (the target),
# "what did X acquire?" names X as the acquirer (the buyer); retrieval must honor the direction or
# it inverts the relation.
_ACQUIRE_VERB_RE = re.compile(
    r"\b(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b", re.IGNORECASE
)
_ACTIVE_ACQ_PREDICATE = (
    r"(?:has|have|had)\s+(?:acquired|bought|purchased|taken\s*over)"
    r"|(?:is|are|was|were)\s+(?:acquiring|buying|taking\s*over|(?:the\s+)?acquirers?)"
    r"|acquired|acquires|bought|buys|purchased|purchases|took\s*over|takes?\s*over"
)
# "who acquired X?": the interrogative is the SUBJECT of the predicate, so X is the target. The
# predicate must follow it directly, which is what separates it from "who did X acquire?" (X = buyer).
_WHO_ACQUIRED_RE = re.compile(
    rf"\b(?:who|which\s+(?:compan(?:y|ies)|firms?|business(?:es)?))\s+(?:{_ACTIVE_ACQ_PREDICATE})\b",
    re.IGNORECASE,
)
# Longest connective span the relation patterns scan between their two fixed ends; real multi-clause
# queries put one in the tens of characters, so the bound is invisible on real input. It must be
# bounded because the input is the caller's query and an unbounded `.*?` gap is rescanned from each
# literal start position, making one search O(n^2) in query length. A gap that was `.+?` must stay
# one-or-more (see `_ALL_OF_RE`): a zero floor widens a matcher whose prefix ends in a consumable
# character rather than a zero-width `\b`.
_MAX_CONNECTIVE_SPAN = 200


_ACQUIRED_BY_RE = re.compile(
    rf"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b[^.?!]{{0,{_MAX_CONNECTIVE_SPAN}}}?"
    rf"\bby\b[^.?!]{{0,{_MAX_CONNECTIVE_SPAN}}}?\b(who|whom)\b",
    re.IGNORECASE,
)
_BUYER_AUX_RE = re.compile(
    rf"\b(?:who|whom|what|which\s+\w+)\b[^.?!]{{0,{_MAX_CONNECTIVE_SPAN}}}?"
    rf"\b(?:did|does|do|has|have|had|is|are|was|were)\b"
    rf"[^.?!]{{0,{_MAX_CONNECTIVE_SPAN}}}?"
    rf"\b(acquir\w+|bought|buys?|buying|purchas\w+|take\s*over|taken\s*over|took\s*over|takeover)\b",
    re.IGNORECASE,
)
_BUYER_TRAILING_RE = re.compile(
    rf"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b.{{0,{_MAX_CONNECTIVE_SPAN}}}"
    rf"\b(what|whom|who)\b",
    re.IGNORECASE,
)


def acquisition_relation(query: str) -> str | None:
    """``'target'`` when the named company was acquired ("who acquired X?"), ``'buyer'`` when it did
    the acquiring ("what did X acquire?"), or ``None`` when the query carries no such intent."""
    if not _ACQUIRE_VERB_RE.search(query):
        return None
    # Target patterns are the stricter ones, so they run first: "who has acquired X?" is a target
    # query even though the looser buyer pattern matches its shape too.
    if _WHO_ACQUIRED_RE.search(query) or _ACQUIRED_BY_RE.search(query):
        return "target"
    if _BUYER_AUX_RE.search(query) or _BUYER_TRAILING_RE.search(query):
        return "buyer"
    return None


_COMPARE_RE = re.compile(
    r"\b(versus|vs\.?|compare|compared to|compared with|"
    r"differences? between|contrast|how do(?:es)? .* compare)\b",
    re.IGNORECASE,
)
_BOTH_AND_RE = re.compile(
    rf"\bboth\b.{{0,{_MAX_CONNECTIVE_SPAN}}}?\band\b", re.IGNORECASE
)
_ALL_OF_RE = re.compile(
    # Lower bound 1, not 0: this gap must stay one-or-more. `\ball ` ends in a literal space, so a
    # zero-length gap lets `\b(?:and|with)\b` match right after it, flipping search("all and") to
    # True -- which flips detect_multi_entity from 'comparison' to 'intersection'. The other four
    # relation patterns end in a zero-width `\b`, so a zero floor cannot widen them.
    rf"\ball (?:of )?.{{1,{_MAX_CONNECTIVE_SPAN}}}?\b(?:and|with)\b",
    re.IGNORECASE,
)


def _strip_entities(text: str, entities: list[str]) -> str:
    s = text
    for e in entities:
        if not e:
            continue
        s = re.sub(rf"\b{re.escape(e)}\b", " ", s, flags=re.IGNORECASE)
    return s


def _multi_entity_scaffold(query: str, entities: list[str]) -> str:
    """Topical remainder of a multi-entity query once entity names and connectives are removed
    ('compare funding of SoftBank and Tiger Global' -> 'funding'); used to build a per-entity
    retrieval query."""
    s = _strip_entities(query.lower(), entities)
    s = _COMPARE_RE.sub(" ", s)
    s = _BOTH_AND_RE.sub(" ", s)
    s = _ALL_OF_RE.sub(" ", s)
    s = re.sub(r"\b(?:both|all|and|with|the|of|for|to|in|by|from|between)\b", " ", s)
    s = re.sub(r"[\s-]+", " ", s).strip()
    return s


class MultiEntityQuery:
    """A query spanning two or more entities, as a comparison or an intersection. ``scaffold`` is
    the topic left after stripping entities and connectives."""

    def __init__(self, mode: str, entities: list[str], scaffold: str):
        self.mode = mode
        self.entities = entities
        self.scaffold = scaffold


def detect_multi_entity(query: str) -> "MultiEntityQuery | None":
    """Detect a comparison or intersection query over two or more entities.

    Needs BOTH a cue and two or more named entities; a single-entity query returns None so the
    caller's normal path runs, and an intersection cue wins over a comparison cue."""
    from app.rerank_boost import extract_entities

    entities = extract_entities(query)
    if len(entities) < 2:
        return None
    is_intersection = bool(_BOTH_AND_RE.search(query) or _ALL_OF_RE.search(query))
    is_comparison = bool(_COMPARE_RE.search(query))
    if not (is_intersection or is_comparison):
        return None
    mode = "intersection" if is_intersection else "comparison"
    scaffold = _multi_entity_scaffold(query, entities)
    return MultiEntityQuery(mode=mode, entities=entities, scaffold=scaffold)
