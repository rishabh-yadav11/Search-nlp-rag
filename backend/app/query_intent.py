import calendar
import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

# 'today' follows the Indian calendar, not the host's: between 00:00 and 05:30
# IST the UTC date is still the previous day, which would resolve the wrong
# year around New Year on a UTC server. `tzdata` is pinned in requirements.txt
# because stdlib zoneinfo falls back to it when the host has no system tz
# database; without it this import raises ZoneInfoNotFoundError at boot and
# there is deliberately no degraded UTC-offset fallback.
_IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    """The current instant as an aware UTC datetime; the module's single clock
    seam, which tests patch to freeze time."""
    return datetime.now(UTC)


def _today() -> date:
    """Today's date in Asia/Kolkata; the conversion happens here so freezing
    ``_now`` still exercises the Indian-calendar resolution."""
    return _now().astimezone(_IST).date()


def _current_year() -> int:
    """The current Indian year, computed per call so it stays correct across a
    calendar-year boundary in a long-running process."""
    return _today().year

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_YEAR_SPAN_RE = re.compile(
    r"\b(20\d{2}|19\d{2})\s*(?:-|to|through|and)\s*(20\d{2}|19\d{2})\b", re.IGNORECASE
)
# Short year span: '2024-25' -> years 2024 and 2025 (same century). The trailing
# \b on the 2-digit year keeps it from matching inside a 4-digit year.
_YEAR_SPAN_SHORT_RE = re.compile(
    r"\b(20\d{2}|19\d{2})\s*(?:-|to|through)\s*(\d{2})\b", re.IGNORECASE
)
_LAST_YEAR_RE = re.compile(r"\b(?:the\s+)?last\s+year\b|\bprevious\s+year\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)
_FLASHBACK_RE = re.compile(r"\bflashback\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
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
# Tens+unit compounds ('fortyfive', 'twentyone') plus the plain tens ('forty')
# for standalone use.
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
# Superlative/aggregation hints beyond the plain list words: the user wants a
# ranked/aggregated answer rather than a single fact. "most" only counts when it
# precedes an aggregation noun, since bare "most" is far too common; "least" is a
# genuine superlative but the threshold phrase "at least" must NOT be one, hence
# the negative-lookbehind on "at ".
_SUPERLATIVE_ALT = r"biggest|largest|highest|greatest|maximum|smallest|lowest|(?<!at\s)least"
_AGG_NOUN_ALT = (
    r"active|funded|funding|invested|investing|investments?|valuable|valued|"
    r"profitable|raised|successful|mentioned|cited|covered|popular|influential|"
    r"deal(s|making)?|acquisitive"
)
_SUPERLATIVE_RE = re.compile(
    rf"\b({_SUPERLATIVE_ALT})\b|\bmost\s+({_AGG_NOUN_ALT})\b", re.IGNORECASE
)
# Filler/time words dropped when extracting a bare topic from a query.
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
# A span of months: 'jan-march', 'january 2025 to march 2025', 'between january
# and march'. Each month may carry its own year.
_MONTH_SPAN_RE = re.compile(
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?\s*(?:-|to|through|and)\s*"
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?",
    re.IGNORECASE,
)
# month(+optional year) with optional filler words: "january 2025", "in jan"
_MONTH_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALT})\b[^.\d]*(?:\b(20\d{{2}}|19\d{{2}})\b)?",
    re.IGNORECASE,
)

# Quarter references: 'Q1 2025', 'Q1-2025', "Q1'25", 'first quarter of 2025'.
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

# Fiscal-year references: 'FY25', 'FY 25', "FY'25", 'FY2024-25', 'FY 2024 to 2025',
# 'fiscal year 2025'. An Indian FY ending in year N spans Apr (N-1) to Mar N.
# The groups are NAMED so `_fiscal_range` can pick whichever alternative
# matched by name; this one grammar is shared with `_strip_time_tokens`, so a
# syntax change cannot drift between resolving a range and stripping the token.
# The last branch takes `\s*` so the fused 'fiscal2020' spelling resolves AND
# strips as a fiscal year; requiring whitespace (`\s+`) would drop the token from
# the retrieval query while the date filter still fired.
_FY_RE = re.compile(
    r"\bfy\s*(?P<fy_range_start>(?:19|20)?\d{2})\s*(?:-|to|through)\s*(?P<fy_range_end>(?:19|20)?\d{2})\b"
    r"|\bfy\s*'?(?P<fy_single>(?:19|20)?\d{2})\b"
    r"|\bfiscal\s+year\s*(?P<fiscal_year>(?:19|20)?\d{2})\b"
    r"|\bfiscal\s*(?P<fiscal_bare>(?:19|20)?\d{2})\b",
    re.IGNORECASE,
)

# A year naming a historical event is a topic reference, not a publication-date
# filter: 'the 2008 crisis' should surface retrospectives written later, so the
# auto date filter is suppressed for such phrases.
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

# Chart/table request filler: 'make a table of', 'show me a bar chart of'. These
# words describe the requested OUTPUT format, not the topic, so they must be
# stripped from the retrieval/rerank query or the embedding match is diluted
# ('make a table of top 15 deals' would retrieve on 'make a table'). Table nouns
# exclude the 'table tennis' collocation and need an article or
# format/form/view suffix in trailing position: bare 'in table' is a common-noun
# phrase, not a view request.
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


# Content-type intent modifiers: 'interviews with X', 'founders of Y',
# 'competitors of Z', 'appointments'. These name the kind of article the user
# wants, distinct from its dealtype/industry. The bare modifier maps to a
# canonical content-type keyword; main.resolve_content_type then promotes it to a
# real facet value from the live `content_type` vocabulary (exact, then substring),
# so an unknown corpus degrades to no filter rather than a bogus value. Aliases
# are matched whole-word, longest-first, so 'competitors' beats 'compete' and
# 'founder' beats 'found'.
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
    """The canonical content-type keyword implied by ``query`` (e.g. 'interviews
    with X' -> 'interview'), or None. Resolution to a real facet value happens in
    main.resolve_content_type against the live vocabulary."""
    q = query.lower()
    for alias, kw in sorted(_CONTENT_TYPE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            return kw
    return None


def _is_chart_request(text: str) -> bool:
    """True when the query contains a chart/table request phrase, so its filler
    words can be stripped from the retrieval topic."""
    return bool(_CHART_LEAD_RE.search(text) or _CHART_TRAIL_RE.search(text))


def _strip_chart_filler(text: str) -> str:
    """Remove chart/table request filler words, leaving the bare topic text."""
    s = _CHART_LEAD_RE.sub(" ", text)
    s = _CHART_TRAIL_RE.sub(" ", s)
    return s


def _year_is_event_reference(query: str, year: int) -> bool:
    """True when ``year`` appears inside a historical-event phrase ('2008 crisis'),
    making it a topic reference rather than a publication-date filter."""
    for pattern, year_group in ((_EVENT_YEAR_RE, 1), (_REV_EVENT_YEAR_RE, 2)):
        for m in pattern.finditer(query):
            if int(m.group(year_group)) == year:
                return True
    return False


def _full_year(y: int, near: int) -> int:
    """Expand a 2-digit year to 4 digits near ``near`` using a 50-year pivot: the
    year is placed in ``near``'s century, then rolled a century back or forward
    when it lands more than 50 years ahead of or behind ``near`` ('99' near 2024
    -> 1999, '20' near 2071 -> 2020)."""
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
    """(from_date, to_date) ISO strings for a month or month span in the query,
    e.g. 'january 2025' -> ('2025-01-01', '2025-01-31'). Year defaults to the
    current year when not given. None when no month is mentioned."""
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
    """A span of months as (from_date, to_date), or None. Handles 'jan-march',
    'january to march 2025', 'january 2025 to march 2025', and 'january and
    february'. A reverse span (e.g. 'may to march') crosses a year boundary; a
    single year anchors the month it is written next to ('dec to jan 2024' ->
    2023-12..2024-01)."""
    m = _MONTH_SPAN_RE.search(query)
    if not m:
        return None
    m1, y1, y2 = _MONTHS[m.group(1)], m.group(2), m.group(4)
    m2 = _MONTHS[m.group(3)]
    crosses_year = m1 > m2
    if y1 and y2:
        # Two explicit years: honor each side's year exactly.
        start_year, end_year = int(y1), int(y2)
    elif y2:
        # A year attached to the END month anchors the end, so a
        # boundary-crossing span starts the year BEFORE it ('dec to jan 2024').
        end_year = int(y2)
        start_year = end_year - 1 if crosses_year else end_year
    elif y1:
        # A year attached to the START month anchors the start, so a
        # boundary-crossing span ends the year AFTER it ('dec 2023 to jan').
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
    """(from_date, to_date) for a quarter reference ('Q1 2025', 'first quarter
    of 2025'), or None. Year defaults to the current year when not given."""
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
    """(from_date, to_date) for a fiscal-year reference ('FY25', 'FY 2024-25',
    'fiscal year 2025'), or None. FY ending in year N spans Apr (N-1) to Mar N."""
    q = query.lower()
    # The shared `_FY_RE` grammar in three priority tiers: an explicit span,
    # then a bare 'fy N', then 'fiscal year N'/'fiscal N'. A tier wins wherever
    # it appears, so matches are scanned in full rather than taking the leftmost
    # one -- otherwise 'fiscal 2025 and fy 2020-2021' would resolve 2025.
    matches = list(_FY_RE.finditer(q))
    span = next((m for m in matches if m.group("fy_range_start")), None)
    if span:
        y1 = _full_year(int(span.group("fy_range_start")), _current_year())
        y2 = _full_year(int(span.group("fy_range_end")), y1)
        # An FY span lists start and end years ('FY 2024-25' -> FY2024-2025).
        # Written end-first ('fy 2025-24') the larger number is still the ending
        # year, so take min/max rather than a wrong century rollover. A
        # multi-year span ('fy 2020-2025') opens f"{start}-04-01" and closes
        # f"{end}-03-31", which is only valid while start < end.
        start = min(y1, y2)
        end = max(y1, y2)
        # start == end means the span names one year twice ('fy 25-25'), i.e. a
        # single fiscal year: fall back to end - 1 so the window stays valid
        # instead of matching zero rows.
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


# Rolling-window recency phrases ("this week", "today", "past 3 days") resolve to a
# concrete recent date range so the filter excludes old evergreen articles. Number-
# bearing forms capture (count, unit) in groups 1-2 ("past 3 days") or 3-4 ("2 weeks
# ago").
_RECENCY_WINDOW_RE = re.compile(
    r"\b(?:today|"
    r"this\s+week|past\s+week|last\s+week|"
    r"this\s+month|past\s+month|last\s+month|"
    r"(?:past|last)\s+(\d+)\s*(day|days|week|weeks|month|months)|"
    r"(\d+)\s*(day|days|week|weeks|month|months)\s*ago)\b",
    re.IGNORECASE,
)
# Soft recency/freshness signals weight recency in ranking rather than filtering.
# 'current'/'upcoming' are excluded: 'current account' is a finance topic, and
# 'upcoming' points at future events the corpus may not yet cover.
_RECENCY_INTENT_RE = re.compile(
    r"\b(latest|recent|newest|freshest|fresh|lately|breaking|of\s+late)\b",
    re.IGNORECASE,
)

_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


def _days_ago_iso(days: int) -> str:
    """ISO date ``days`` before today, on the module's Indian-calendar 'now' so
    the resolution matches a rolling recency window."""
    return (_today() - timedelta(days=days)).isoformat()


def _month_start_iso() -> str:
    """ISO date of the first day of the current (Indian) month, anchoring 'this
    month' to the calendar boundary so prior-month articles are excluded."""
    return _today().replace(day=1).isoformat()


def _week_start_iso() -> str:
    """ISO date of Monday of the current week (Indian 'now'), anchoring 'this
    week' to the calendar boundary so last week's articles are excluded."""
    t = _today()
    return (t - timedelta(days=t.weekday())).isoformat()


def extract_recency_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) ISO strings for a rolling recency window in the query
    ('this week', 'today', 'past 3 days'), or None. A hard window filters out old
    evergreen articles; soft recency signals ('latest', 'recent') have no fixed
    window and are left to ``is_recency_intent`` (a ranking weight)."""
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
    # 'this week'/'this month' anchor to the calendar boundary of the current
    # week/month so prior-period articles are excluded; other week/month forms
    # ('past week', 'last month') keep their rolling window semantics.
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
    """Remove rolling-window recency phrases, leaving the bare topic text."""
    return _RECENCY_WINDOW_RE.sub(" ", query).strip()


def strip_recency_intent(query: str) -> str:
    """Remove soft recency/freshness signals from a query's retrieval text. The
    ranking intent (``is_recency_intent``) must still be detected on the
    original query."""
    return _RECENCY_INTENT_RE.sub(" ", query).strip()


def is_recency_intent(query: str) -> bool:
    """True when the query expresses a soft recency/freshness preference ('latest
    news', 'recent funding') with no fixed window, so ranking should weight
    recency. Hard-window phrases ('this week') are filtered separately."""
    return bool(_RECENCY_INTENT_RE.search(query))


def extract_year_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) ISO strings for a time window in the query: a month
    or month span, a fiscal year, a quarter, a year span, an explicit year, or
    'last year'/'this year', in that precedence order. A year naming a historical
    event ('2008 crisis') is NOT a publication-date filter: such queries want
    retrospectives written later, not only articles published that year."""
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
        # A descending span ('2025 to 2024') is the same window end-first;
        # normalize rather than emit an inverted (from > to) window matching
        # nothing.
        start, end = min(y1, y2), max(y1, y2)
        return (f"{start}-01-01", f"{end}-12-31")
    m = _YEAR_SPAN_SHORT_RE.search(q)
    if m:
        start = int(m.group(1))
        end = _full_year(int(m.group(2)), start)
        # A descending short span ('2024-23') is a reversed year span; normalize
        # rather than take a century rollover.
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
    """Convert a 'top N' count phrase ('10', 'ten', 'twenty five') to an int,
    or None when the phrase is not a recognizable count."""
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
    """Rewrite word-form counts after a list hint to digits so the retrieval query
    matches the numeric form: 'top ten ipo' -> 'top 10 ipo'. The literal word
    'ten' pollutes the embedding/rerank match (titles like 'Ten Sports')."""
    def _replace(m: re.Match) -> str:
        n = _top_n_to_int(m.group(2))
        return f"{m.group(1)} {n}" if n is not None else m.group(0)

    return _TOP_N_RE.sub(_replace, query)


def suggested_top_k(query: str) -> int | None:
    """Suggested top_k from a 'top N' in the query, or a small default for a
    generic top/best or superlative intent so chat fetches enough articles to
    aggregate into a ranked list. None when no list intent."""
    m = _TOP_N_RE.search(query)
    if m:
        n = _top_n_to_int(m.group(2))
        if n is not None:
            return n
    if _TOP_HINT_RE.search(query) or _is_superlative(query):
        return _DEFAULT_LIST_K
    return None


def _is_superlative(query: str) -> bool:
    """True when the query uses a superlative/aggregation phrase, independent of
    any explicit 'top N' count."""
    return bool(_SUPERLATIVE_RE.search(query))


def is_aggregation_intent(query: str) -> bool:
    """True when the query asks for a ranked/aggregated answer over many items, so
    chat must present a ranked top-N with the metric that justifies the ordering
    rather than isolated single items. Also widens the retrieved source set and
    triggers the ranked-list prompt and refusal nudge."""
    return suggested_top_k(query) is not None


def _strip_time_tokens(text: str) -> str:
    """Remove fiscal/quarter/month/year/time filler tokens from a query, leaving
    the bare topical text. Time-word regexes are applied longest-first so a
    compound token ('FY 2024-25', 'jan to march') is removed before its parts."""
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
    """For 'top/best <topic> in <year>' style queries, rewrite to surface the
    year-in-review ('Flashback <year>') articles. Returns (query, changed).

    Month-scoped queries are NOT rewritten: Flashback articles are annual
    roundups, so a month-scoped query should match that month's articles.
    Range/fiscal/quarter queries are not rewritten either -- they want the span's
    own data, not a single annual roundup."""
    if extract_month_range(query) is not None:
        return query, False
    topic = extract_list_topic(query)
    yr = _referenced_year(query)
    if yr is None or topic is None:
        return query, False
    new_q = f"Flashback {yr} {topic}".strip()
    return new_q, new_q != query


def _referenced_year(query: str) -> int | None:
    """The year referenced by the query (explicit, last/this year, or an explicit
    'Flashback <year>' prefix), else None. Range, fiscal-year, and quarter queries
    return None: they span more than a single calendar year (or a sub-year period)
    and must not collapse into one annual roundup."""
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
    """Cleaned retrieval/rerank query for a query scoped by an auto date range that
    is NOT a plain single year, e.g. 'top 15 deals in Q1 2025' -> 'deals'. The
    date filter already scopes the period, so dropping the range words lets the
    embeddings and cross-encoder focus on the topic. None for plain single-year
    queries (they keep their Flashback rewrite) or with no auto date range."""
    if extract_month_range(query) is not None:
        return extract_list_topic(query) or _strip_noise_words(query)
    if _quarter_range(query) is not None or _fiscal_range(query) is not None:
        return extract_list_topic(query) or _strip_noise_words(query)
    if _YEAR_SPAN_RE.search(query) or _YEAR_SPAN_SHORT_RE.search(query):
        return extract_list_topic(query) or _strip_noise_words(query)
    return None


def extract_list_topic(query: str) -> str | None:
    """The bare topic of a top-N query with year/time words removed. None when
    the query is not a top/best/list intent."""
    if _TOP_HINT_RE.search(query) is None and _TOP_N_RE.search(query) is None:
        return None
    stripped = _strip_time_tokens(query)
    stripped = _TOP_N_RE.sub(" ", stripped)
    stripped = _TOP_HINT_RE.sub(" ", stripped)
    topic = re.sub(r"[\s-]+", " ", stripped).strip()
    return topic or None


def _strip_noise_words(query: str) -> str | None:
    """Remove month/year/time filler words, leaving the bare query text."""
    q = _strip_time_tokens(query)
    q = re.sub(r"[\s-]+", " ", q).strip()
    return q or None


# Acquisition relation direction. "who acquired X?" names X as the company that
# WAS acquired (the target), whereas "what did X acquire?" names X as the company
# that DID the acquiring (the buyer). Retrieval must honor this direction so it
# surfaces the right counterpart instead of inverting the relation.
_ACQUIRE_VERB_RE = re.compile(
    r"\b(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b", re.IGNORECASE
)
# An active acquisition predicate: the grammatical subject in front of it is the
# buyer, so "<interrogative> <predicate> X" makes X the target.
_ACTIVE_ACQ_PREDICATE = (
    r"(?:has|have|had)\s+(?:acquired|bought|purchased|taken\s*over)"
    r"|(?:is|are|was|were)\s+(?:acquiring|buying|taking\s*over|(?:the\s+)?acquirers?)"
    r"|acquired|acquires|bought|buys|purchased|purchases|took\s*over|takes?\s*over"
)
# "who acquired X?" / "which company bought X?": the interrogative is the SUBJECT
# of an active acquisition predicate, so the named company X is the target. The
# predicate has to follow the interrogative directly, which keeps "who did X
# acquire?" and "who was acquired by X?" -- where X is the buyer -- from matching.
_WHO_ACQUIRED_RE = re.compile(
    rf"\b(?:who|which\s+(?:compan(?:y|ies)|firms?|business(?:es)?))\s+(?:{_ACTIVE_ACQ_PREDICATE})\b",
    re.IGNORECASE,
)
# Longest connective span the relation patterns below will scan between their two
# fixed ends ("both" ... "and", an acquire verb ... "by" ... "who"). Real
# multi-clause queries put such a span in the tens of characters, so the bound is
# invisible on real input; a query whose connective spans 200+ characters is
# pathological. The gap must be bounded because the input is the caller's query:
# an unbounded `.*?` gap is rescanned from each of the k literal start positions,
# so one search costs O(n^2) in query length. The LOWER bound is per-pattern: a
# pattern whose prefix ends in a consumable character (not a zero-width `\b`)
# needs `{1,N}?`, since a zero floor widens the matcher there. See
# `_ALL_OF_RE` for the one that needs it; the rest take `{0,N}?`.
_MAX_CONNECTIVE_SPAN = 200


_ACQUIRED_BY_RE = re.compile(
    rf"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b[^.?!]{{0,{_MAX_CONNECTIVE_SPAN}}}?"
    rf"\bby\b[^.?!]{{0,{_MAX_CONNECTIVE_SPAN}}}?\b(who|whom)\b",
    re.IGNORECASE,
)
# "who did X acquire?" / "which company was acquired by X?": the interrogative
# stands for the counterpart, so the named company X is the buyer.
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
    """Infer the acquisition relation direction implied by ``query``: ``'target'``
    when the named company was acquired ("who acquired X?"), ``'buyer'`` when it
    did the acquiring ("what did X acquire?"), else None."""
    if not _ACQUIRE_VERB_RE.search(query):
        return None
    # Target patterns are the stricter ones, so they are tested first: "who has
    # acquired X?" is a target query even though the looser buyer pattern matches
    # its "who ... has ... acquired" shape too.
    if _WHO_ACQUIRED_RE.search(query) or _ACQUIRED_BY_RE.search(query):
        return "target"
    if _BUYER_AUX_RE.search(query) or _BUYER_TRAILING_RE.search(query):
        return "buyer"
    return None


# Comparison cues: weighing two named entities against each other ("X vs Y").
_COMPARE_RE = re.compile(
    r"\b(versus|vs\.?|compare|compared to|compared with|"
    r"differences? between|contrast|how do(?:es)? .* compare)\b",
    re.IGNORECASE,
)
# Intersection cues: what is shared across entities ("backed by both A and B").
_BOTH_AND_RE = re.compile(
    rf"\bboth\b.{{0,{_MAX_CONNECTIVE_SPAN}}}?\band\b", re.IGNORECASE
)
_ALL_OF_RE = re.compile(
    # Lower bound 1, not 0: `\ball ` ends in a literal space, so a zero-length gap
    # lets `\b(?:and|with)\b` match immediately after it -- `search("all and")`
    # would flip False -> True. That is not cosmetic: it changes the
    # `_multi_entity_scaffold` output and flips `detect_multi_entity` from
    # 'comparison' to 'intersection', changing per-entity retrieval. The other
    # patterns are unaffected by a zero floor because their prefixes end in a
    # zero-width `\b` (no word character can follow).
    rf"\ball (?:of )?.{{1,{_MAX_CONNECTIVE_SPAN}}}?\b(?:and|with)\b",
    re.IGNORECASE,
)


def _strip_entities(text: str, entities: list[str]) -> str:
    """Remove each entity mention (word-bounded, case-insensitive) from ``text``."""
    s = text
    for e in entities:
        if not e:
            continue
        s = re.sub(rf"\b{re.escape(e)}\b", " ", s, flags=re.IGNORECASE)
    return s


def _multi_entity_scaffold(query: str, entities: list[str]) -> str:
    """The topical remainder of a multi-entity query once the entity names and the
    comparison/intersection connectives are removed, used to build a per-entity
    retrieval query that keeps the topic while swapping in a single entity."""
    s = _strip_entities(query.lower(), entities)
    s = _COMPARE_RE.sub(" ", s)
    s = _BOTH_AND_RE.sub(" ", s)
    s = _ALL_OF_RE.sub(" ", s)
    s = re.sub(r"\b(?:both|all|and|with|the|of|for|to|in|by|from|between)\b", " ", s)
    s = re.sub(r"[\s-]+", " ", s).strip()
    return s


class MultiEntityQuery:
    """A query over two or more entities, either a comparison or an intersection.
    ``scaffold`` is the topic left after stripping entities and connectives, used
    to build a per-entity retrieval query."""

    def __init__(self, mode: str, entities: list[str], scaffold: str):
        self.mode = mode
        self.entities = entities
        self.scaffold = scaffold


def detect_multi_entity(query: str) -> "MultiEntityQuery | None":
    """Detect a comparison or intersection query over two or more entities.

    Returns a :class:`MultiEntityQuery` when the query names at least two entities
    AND carries a comparison or intersection cue, else None so single-entity
    queries take the caller's normal retrieval path."""
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
