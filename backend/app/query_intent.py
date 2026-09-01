import calendar
import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

# The app serves Indian news, so 'today' follows the Indian calendar, not the
# host's: between 00:00 and 05:30 IST the UTC date is still the previous day,
# which would resolve the wrong year around New Year on a UTC server.
# `tzdata` is pinned in requirements.txt: stdlib zoneinfo falls back to that
# package when the host has no system tz database. That pin -- not the stdlib
# alone -- is what makes this zone resolvable on a slim container. If it is not
# installed (an image built from older requirements, or a deploy that skips
# `pip install -r requirements.txt`), this import raises ZoneInfoNotFoundError
# and the app dies at boot; there is deliberately no degraded UTC-offset
# fallback, because deployments always install requirements.txt.
_IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    """The current instant, as an aware UTC datetime. This is the module's
    single clock seam: tests freeze time by patching this helper
    (``monkeypatch.setattr(query_intent, "_now", lambda: frozen)``) rather than
    the imported ``datetime`` class, so they keep working if this changes."""
    return datetime.now(UTC)


def _today() -> date:
    """Today's date in Asia/Kolkata (UTC+05:30). Single source of 'now' for
    this module; the conversion from UTC happens here so freezing ``_now``
    still exercises the Indian-calendar resolution."""
    return _now().astimezone(_IST).date()


def _current_year() -> int:
    """The current Indian year, computed at call time so resolution stays
    correct across a calendar-year boundary in a long-running process."""
    return _today().year

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
# Full year span: '2024 to 2025', '2023-2025', '2023 through 2025'.
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
# Word-form counts for 'top ten deals' (mirrors the digit form). Built
# longest-first so 'fourteen' matches before 'four'.
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
# Concatenated tens+unit compounds ('fortyfive', 'twentyone') plus the plain
# tens ('forty') for standalone use.
for _tens_word, tens_val in _TENS.items():
    _NUMBER_WORDS[_tens_word] = tens_val
    for _unit_word, unit_val in _UNITS.items():
        _NUMBER_WORDS[f"{_tens_word}{_unit_word}"] = tens_val + unit_val
_NUMBER_WORDS["hundred"] = 100
_NUM_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_WORD_SEP = r"(?:\s+|-|\s+-\s+)"
# 'top 10' or 'top ten', 'top twenty five', 'top twenty-five', 'best ten'.
# Any list-hint word (top/best/leading/biggest/largest) may precede the count.
_TOP_HINT_ALT = r"top|best|leading|biggest|largest"
_TOP_N_RE = re.compile(
    rf"\b({_TOP_HINT_ALT})\s+((?:\d{{1,4}})|(?:(?:{_NUM_WORD_ALT})(?:{_WORD_SEP}(?:{_NUM_WORD_ALT}))*))\b",
    re.IGNORECASE,
)
_TOP_HINT_RE = re.compile(r"\b(best|leading|biggest|largest|top)\b", re.IGNORECASE)
_DEFAULT_LIST_K = 10
# Superlative/aggregation hints beyond the plain list words. These signal the
# user wants a ranked/aggregated answer over many items ("biggest funding
# rounds", "highest valued startups", "most active investors") rather than a
# single fact. "most" only counts when it precedes an aggregation noun
# (active/funded/valued/...): bare "most" is far too common ("most of the time").
# "least" is a genuine superlative ("least funded"), but the common threshold
# phrase "at least" must NOT be treated as one, so we negative-lookbehind "at ".
_SUPERLATIVE_ALT = r"biggest|largest|highest|greatest|maximum|smallest|lowest|(?<!at\s)least"
_AGG_NOUN_ALT = (
    r"active|funded|funding|invested|investing|investments?|valuable|valued|"
    r"profitable|raised|successful|mentioned|cited|covered|popular|influential|"
    r"deal(s|making)?|acquisitive"
)
_SUPERLATIVE_RE = re.compile(
    rf"\b({_SUPERLATIVE_ALT})\b|\bmost\s+({_AGG_NOUN_ALT})\b", re.IGNORECASE
)
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
_MONTH_ALT = (
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_MONTH_RE = re.compile(rf"\b({_MONTH_ALT})\b", re.IGNORECASE)
# A span of months: 'jan-march', 'january to march 2025', 'january 2025 to march
# 2025', 'between january and march'. Each month may carry its own year.
_MONTH_SPAN_RE = re.compile(
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?\s*(?:-|to|through|and)\s*"
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?",
    re.IGNORECASE,
)
# month(+optional year) with optional filler words: "january 2025", "of month january 2025", "in jan"
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
_FY_RE = re.compile(
    r"\bfy\s*((?:19|20)?\d{2})\s*(?:-|to|through)\s*((?:19|20)?\d{2})\b"
    r"|\bfy\s*'?((?:19|20)?\d{2})\b"
    r"|\bfiscal\s+year\s*((?:19|20)?\d{2})\b"
    r"|\bfiscal\s+((?:19|20)?\d{2})\b",
    re.IGNORECASE,
)

# A year that names a historical event is a topic reference, not a
# publication-date filter: 'the 2008 crisis' should surface retrospectives
# written later, so the auto date filter is suppressed for such phrases.
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
# '2008 (financial) crisis', 'the 2008 global financial crisis'
_EVENT_YEAR_RE = re.compile(
    rf"\b((?:19|20)\d{{2}})\s+(?:\w+\s+){{0,2}}({_EVENT_NOUN_ALT})s?\b", re.IGNORECASE
)
# 'financial crisis of 2008', 'the crisis in 2008'
_REV_EVENT_YEAR_RE = re.compile(
    rf"\b(?:\w+\s+){{0,2}}({_EVENT_NOUN_ALT})s?\s+(?:of|in)\s+((?:19|20)\d{{2}})\b", re.IGNORECASE
)

# Chart/table request filler: 'make a table of', 'show me a bar chart of',
# 'create a pie chart for', 'top deals as a table'. These words describe the
# requested OUTPUT format, not the topic, so they must be stripped from the
# retrieval/rerank query or the embedding match is diluted ('make a table of
# top 15 deals' would otherwise retrieve on 'make a table' instead of 'deals').
_CHART_VERB = r"(?:show|draw|make|create|give|build|plot|display|present)"
_CHART_TYPE = r"(?:bar|line|pie|column|area|pictogram|pictograph)?\s*"
_CHART_NOUN = r"(?:chart|graph|plot|diagram|table|pictogram|pictograph)"
_CHART_LEAD_RE = re.compile(
    rf"\b(?:{_CHART_VERB})\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?"
    rf"{_CHART_TYPE}{_CHART_NOUN}\b(?:\s+(?:of|for|on|about|regarding)\b)?",
    re.IGNORECASE,
)
_CHART_TRAIL_RE = re.compile(
    rf"\b(?:as|in|into|using)\s+(?:a\s+|an\s+|the\s+)?"
    rf"{_CHART_TYPE}{_CHART_NOUN}\b(?:\s+form\b)?",
    re.IGNORECASE,
)


# Content-type intent modifiers: 'interviews with X', 'founders of Y',
# 'competitors of Z', 'appointments'. These name the kind of article the user
# wants, distinct from its dealtype/industry. The bare modifier maps to a
# canonical content-type keyword; main.resolve_content_type then promotes it to a
# real facet value from the live `content_type` vocabulary (exact, then substring),
# so an unknown corpus degrades to no filter (current behavior) rather than a
# bogus value. Aliases are matched whole-word, longest-first, so 'competitors'
# beats 'compete' and 'founder' beats 'found'.
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
    with X' -> 'interview', 'founders of Y' -> 'founder'), or None when the query
    carries no content-type modifier. Resolution to a real facet value happens in
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
    """True when ``year`` appears in the query inside a historical-event phrase
    (e.g. '2008 crisis' or 'financial crisis of 2008'), meaning it is a topic
    reference rather than a publication-date filter."""
    for pattern, year_group in ((_EVENT_YEAR_RE, 1), (_REV_EVENT_YEAR_RE, 2)):
        for m in pattern.finditer(query):
            if int(m.group(year_group)) == year:
                return True
    return False


def _full_year(y: int, near: int) -> int:
    """Expand a 2-digit year to 4 digits near ``near`` (e.g. '25' near 2024 ->
    2025), handling century rollover ('99' near 2024 -> 1999).

    A 50-year pivot is used: the 2-digit year is placed in ``near``'s century,
    then rolled back a century if it falls more than 50 years ahead of ``near``
    (e.g. '99' -> 2099 -> 1999) or forward a century if it falls more than 50
    years behind (e.g. '20' near 2071 -> 1920 -> 2020)."""
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
    """Return (from_date, to_date) ISO strings for a month or span of months in
    the query, e.g. 'january 2025' -> ('2025-01-01', '2025-01-31') and
    'january to march 2025' -> ('2025-01-01', '2025-03-31'). Year defaults to
    the current year when not given. None when no month is mentioned."""
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
    2023-12..2024-01, 'dec 2023 to jan' -> 2023-12..2024-01)."""
    m = _MONTH_SPAN_RE.search(query)
    if not m:
        return None
    m1, y1, y2 = _MONTHS[m.group(1)], m.group(2), m.group(4)
    m2 = _MONTHS[m.group(3)]
    crosses_year = m1 > m2
    if y1 and y2:
        # Two explicit years: honor each side's year exactly (e.g. 'dec 2023 to
        # jan 2024' -> start 2023-12, end 2024-01).
        start_year, end_year = int(y1), int(y2)
    elif y2:
        # One explicit year attached to the END month (e.g. 'dec to jan 2024'):
        # it anchors the end, so a boundary-crossing span starts the year BEFORE
        # it, not a year after (2023-12 to 2024-01).
        end_year = int(y2)
        start_year = end_year - 1 if crosses_year else end_year
    elif y1:
        # One explicit year attached to the START month (e.g. 'dec 2023 to
        # jan'): it anchors the start, so a boundary-crossing span ends the year
        # after it (2023-12 to 2024-01).
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
    m = re.search(
        r"\bfy\s*((?:19|20)?\d{2})\s*(?:-|to|through)\s*((?:19|20)?\d{2})\b", q
    )
    if m:
        y1 = _full_year(int(m.group(1)), _current_year())
        y2 = _full_year(int(m.group(2)), y1)
        # An FY span lists the start and end years (e.g. 'FY 2024-25' ->
        # FY2024-2025). When written end-first ('fy 2025-24') the larger
        # number is still the ending year, so take min/max rather than a
        # wrong century rollover. `start` is the first fiscal year of the
        # span and `end` the year the last one closes, so a multi-year span
        # ('fy 2020-2025') spans the whole range: Apr (start) to Mar (end);
        # that window starts on f"{start}-04-01" and closes on
        # f"{end}-03-31", so it is only valid while start < end.
        start = min(y1, y2)
        end = max(y1, y2)
        # min/max guarantees start <= end, so start == end is the sole
        # inverted-window case. It means the span names one year twice
        # ('fy 25-25', 'fy 2025-25', 'fy 2025 to 2025'), which is a single
        # fiscal year, not a span: fall back to end - 1 as the start so the
        # window stays valid instead of matching zero rows.
        if start == end:
            start = end - 1
        return (f"{start}-04-01", f"{end}-03-31")
    m = re.search(r"\bfy\s*'?((?:19|20)?\d{2})\b", q)
    if m:
        end = _full_year(int(m.group(1)), _current_year())
        return (f"{end - 1}-04-01", f"{end}-03-31")
    m = re.search(r"\bfiscal(?:\s+year)?\s*((?:19|20)?\d{2})\b", q)
    if m:
        end = _full_year(int(m.group(1)), _current_year())
        return (f"{end - 1}-04-01", f"{end}-03-31")
    return None


# Rolling-window recency phrases ("this week", "today", "past 3 days") resolve to a
# concrete recent date range so the filter excludes old evergreen articles. Softer
# recency signals ("latest", "recent") have no fixed window and are handled by
# ``is_recency_intent`` (a ranking weight) instead. Number-bearing forms capture
# (count, unit) in groups 1-2 ("past 3 days") or 3-4 ("2 weeks ago").
_RECENCY_WINDOW_RE = re.compile(
    r"\b(?:today|"
    r"this\s+week|past\s+week|last\s+week|"
    r"this\s+month|past\s+month|last\s+month|"
    r"(?:past|last)\s+(\d+)\s*(day|days|week|weeks|month|months)|"
    r"(\d+)\s*(day|days|week|weeks|month|months)\s*ago)\b",
    re.IGNORECASE,
)
# Soft recency/freshness signals express a preference for recent articles without
# naming a fixed window: they weight recency in ranking rather than filtering.
# 'current'/'upcoming' are excluded: 'current account' is a finance topic, and
# 'upcoming' points at future events the corpus may not yet cover.
_RECENCY_INTENT_RE = re.compile(
    r"\b(latest|recent|newest|freshest|fresh|lately|breaking|of\s+late)\b",
    re.IGNORECASE,
)

_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


def _days_ago_iso(days: int) -> str:
    """ISO date ``days`` before today (Asia/Kolkata). Used for rolling recency
    windows so the resolution matches the module's Indian-calendar 'now'."""
    return (_today() - timedelta(days=days)).isoformat()


def _month_start_iso() -> str:
    """ISO date of the first day of the current (Indian) month. Used to anchor
    'this month' to the calendar boundary so prior-month articles are excluded."""
    return _today().replace(day=1).isoformat()


def _week_start_iso() -> str:
    """ISO date of Monday of the current week (Indian 'now'). Used to anchor
    'this week' to the calendar boundary so last week's articles are excluded."""
    t = _today()
    return (t - timedelta(days=t.weekday())).isoformat()


def extract_recency_range(query: str) -> tuple[str, str] | None:
    """(from_date, to_date) ISO strings for a rolling recency window in the
    query ('this week', 'this month', 'today', 'past 3 days', '2 weeks ago'), or
    None. A hard window is applied so old evergreen articles are filtered out;
    soft recency signals ('latest', 'recent') have no fixed window and are left
    to ``is_recency_intent`` (ranking weight) instead."""
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
    """Remove rolling-window recency phrases ('this week', 'today', 'past 3 days')
    from a query, leaving the bare topic text for retrieval."""
    return _RECENCY_WINDOW_RE.sub(" ", query).strip()


def strip_recency_intent(query: str) -> str:
    """Remove soft recency/freshness signals ('latest', 'recent', 'fresh') from a
    query, leaving the bare topic text for retrieval. The recency intent for
    ranking (``is_recency_intent``) must be detected on the original query, so
    this only affects retrieval text, never intent detection."""
    return _RECENCY_INTENT_RE.sub(" ", query).strip()


def is_recency_intent(query: str) -> bool:
    """True when the query expresses a soft recency/freshness preference ('latest
    news', 'recent funding', 'fresh updates') with no fixed window. Such queries
    should weight recency in ranking so recent articles outrank old evergreen
    ones. Hard-window phrases ('this week') are filtered separately and are not
    flagged here."""
    return bool(_RECENCY_INTENT_RE.search(query))


def extract_year_range(query: str) -> tuple[str, str] | None:
    """Return (from_date, to_date) ISO strings for a time window mentioned in the
    query: a specific month or month span ('january 2025', 'jan-march'), a fiscal
    year ('FY25'), a quarter ('Q1 2025'), a year span ('2024-25', '2023 to
    2025'), an explicit year, or 'last year'/'this year'. Month spans take
    precedence, then fiscal, then quarter, then year span.

    An explicit year that names a historical event ('2008 crisis', 'financial
    crisis of 2008') is NOT treated as a publication-date filter: such queries
    want retrospectives written later, not only articles published that year."""
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
        # A descending span ('2025 to 2024') is the same window written
        # end-first; normalize to the ascending range instead of emitting an
        # inverted (from > to) date window that matches nothing.
        start, end = min(y1, y2), max(y1, y2)
        return (f"{start}-01-01", f"{end}-12-31")
    m = _YEAR_SPAN_SHORT_RE.search(q)
    if m:
        start = int(m.group(1))
        end = _full_year(int(m.group(2)), start)
        # A descending short span ('2024-23') is just a reversed year span;
        # normalize to the ascending range instead of a century rollover.
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
    """Rewrite word-form counts after a list hint to digits so the retrieval
    query matches the numeric form exactly: 'top ten ipo' -> 'top 10 ipo'.
    Without this the literal word 'ten' pollutes the embedding/rerank match
    (it can match titles like 'Ten Sports'), while the digit form does not."""
    def _replace(m: re.Match) -> str:
        n = _top_n_to_int(m.group(2))
        return f"{m.group(1)} {n}" if n is not None else m.group(0)

    return _TOP_N_RE.sub(_replace, query)


def suggested_top_k(query: str) -> int | None:
    """Suggested top_k from a 'top N' in the query (digit or word form, e.g.
    'top ten'), or a small default for a generic top/best intent without a
    number. Also defaults for a superlative/aggregation intent ('biggest
    funding rounds', 'most active investors') so chat fetches enough articles
    to aggregate into a ranked list. None when no list intent."""
    m = _TOP_N_RE.search(query)
    if m:
        n = _top_n_to_int(m.group(2))
        if n is not None:
            return n
    if _TOP_HINT_RE.search(query) or _is_superlative(query):
        return _DEFAULT_LIST_K
    return None


def _is_superlative(query: str) -> bool:
    """True when the query uses a superlative/aggregation phrase ('biggest',
    'highest', 'most active'), independent of any explicit 'top N' count."""
    return bool(_SUPERLATIVE_RE.search(query))


def is_aggregation_intent(query: str) -> bool:
    """True when the query asks for a ranked/aggregated answer over many items
    (a 'top N' count, a top/best/leading hint, or a superlative like 'biggest
    funding rounds' / 'most active investors'), so chat must present a ranked
    top-N with the metric that justifies the ordering rather than isolated
    single items. Used to widen the retrieved source set and to trigger the
    ranked-list prompt and refusal nudge."""
    return suggested_top_k(query) is not None


def _strip_time_tokens(text: str) -> str:
    """Remove fiscal/quarter/month/year/time filler tokens from a query, leaving
    the bare topical text. Time-word regexes are applied longest-first so a
    compound token ('FY 2024-25', 'jan to march') is removed before its parts.
    Chart/table request filler ('make a table of') is stripped first when the
    query is a chart request, so the output-format words never reach retrieval."""
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

    Queries that mention a specific MONTH are NOT rewritten: Flashback articles
    are annual roundups, so a month-scoped query should match the month's actual
    articles instead. Range/fiscal/quarter queries are not rewritten either —
    they want the span's own data, not a single annual roundup."""
    if extract_month_range(query) is not None:
        return query, False
    topic = extract_list_topic(query)
    yr = _referenced_year(query)
    if yr is None or topic is None:
        return query, False
    new_q = f"Flashback {yr} {topic}".strip()
    return new_q, new_q != query


def _referenced_year(query: str) -> int | None:
    """The year referenced by the query (explicit, last/this year, or an
    explicit 'Flashback <year>' prefix), else None. Range, fiscal-year, and
    quarter queries return None: they span more than a single calendar year (or
    a sub-year period) and must not collapse into one annual roundup."""
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
    """Cleaned retrieval/rerank query for a query scoped by an auto date range
    that is NOT a plain single year — a month, month span, quarter, fiscal year,
    or year span — e.g. 'top 15 deals in Q1 2025' -> 'deals'. The date filter
    already scopes the period, so dropping the 'top/quarter/fiscal/of/month/year'
    words lets the embeddings and cross-encoder focus on the actual topic. None
    for plain single-year queries (they keep their Flashback rewrite) or for
    queries with no auto date range."""
    if extract_month_range(query) is not None:
        return extract_list_topic(query) or _strip_noise_words(query)
    if _quarter_range(query) is not None or _fiscal_range(query) is not None:
        return extract_list_topic(query) or _strip_noise_words(query)
    if _YEAR_SPAN_RE.search(query) or _YEAR_SPAN_SHORT_RE.search(query):
        return extract_list_topic(query) or _strip_noise_words(query)
    return None


def month_query_topic(query: str) -> str | None:
    """Cleaned retrieval/rerank query for a month-scoped query, e.g.
    'top pharma deals of month january 2025' -> 'pharma deals'. The date filter
    already scopes the month, so dropping the 'top/of/month/year' words lets the
    embeddings and cross-encoder focus on the actual topic. None when the query
    doesn't mention a specific month."""
    if extract_month_range(query) is None:
        return None
    return range_query_topic(query)


def extract_list_topic(query: str) -> str | None:
    """The bare topic of a top-N query with year/time words removed, e.g.
    'top 3 unicorns created in 2025' -> 'unicorns created'. None when the
    query is not a top/best/list intent."""
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
_ACQUIRED_BY_RE = re.compile(
    r"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b[^.?!]*?\bby\b[^.?!]*?\b(who|whom)\b",
    re.IGNORECASE,
)
# "who did X acquire?" / "which company was acquired by X?": the interrogative
# stands for the counterpart, so the named company X is the buyer.
_BUYER_AUX_RE = re.compile(
    r"\b(?:who|whom|what|which\s+\w+)\b[^.?!]*?\b(?:did|does|do|has|have|had|is|are|was|were)\b"
    r"[^.?!]*?\b(acquir\w+|bought|buys?|buying|purchas\w+|take\s*over|taken\s*over|took\s*over|takeover)\b",
    re.IGNORECASE,
)
_BUYER_TRAILING_RE = re.compile(
    r"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b.*\b(what|whom|who)\b",
    re.IGNORECASE,
)


def acquisition_relation(query: str) -> str | None:
    """Infer the acquisition relation direction implied by ``query``.

    Returns ``'target'`` when the named company is the one that WAS acquired
    (e.g. "who acquired X?" -> X is the target), ``'buyer'`` when the named
    company is the one doing the acquiring (e.g. "what did X acquire?" -> X is
    the buyer), or ``None`` when the query has no acquisition-relation intent.
    """
    if not _ACQUIRE_VERB_RE.search(query):
        return None
    # Target patterns are the stricter ones, so they are tested first: "who has
    # acquired X?" is a target query even though the looser buyer pattern would
    # also match its "who ... has ... acquired" shape.
    if _WHO_ACQUIRED_RE.search(query) or _ACQUIRED_BY_RE.search(query):
        return "target"
    if _BUYER_AUX_RE.search(query) or _BUYER_TRAILING_RE.search(query):
        return "buyer"
    return None


# Comparison cues: a question that names two entities and asks to weigh them
# against each other ("X vs Y", "compare A and B", "difference between A and B").
_COMPARE_RE = re.compile(
    r"\b(versus|vs\.?|compare|compared to|compared with|"
    r"differences? between|contrast|how do(?:es)? .* compare)\b",
    re.IGNORECASE,
)
# Intersection cues: a question that wants what is shared across entities
# ("companies backed by both A and B", "deals with all of A, B and C").
_BOTH_AND_RE = re.compile(r"\bboth\b.+?\band\b", re.IGNORECASE)
_ALL_OF_RE = re.compile(r"\ball (?:of )?.+?\b(?:and|with)\b", re.IGNORECASE)


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
    comparison/intersection connectives are removed, e.g. 'compare funding of
    SoftBank and Tiger Global' -> 'funding'. Used to build a per-entity retrieval
    query that keeps the     topic while swapping in a single entity."""
    s = _strip_entities(query.lower(), entities)
    s = _COMPARE_RE.sub(" ", s)
    s = _BOTH_AND_RE.sub(" ", s)
    s = _ALL_OF_RE.sub(" ", s)
    s = re.sub(r"\b(?:both|all|and|with|the|of|for|to|in|by|from|between)\b", " ", s)
    s = re.sub(r"[\s-]+", " ", s).strip()
    return s


class MultiEntityQuery:
    """A query that spans two or more entities, either as a comparison (weigh
    entities against each other) or an intersection (what is shared across all of
    them). ``entities`` are the extracted proper nouns; ``scaffold`` is the topic
    left after stripping the entities and connectives, used to build a per-entity
    retrieval query."""

    def __init__(self, mode: str, entities: list[str], scaffold: str):
        self.mode = mode
        self.entities = entities
        self.scaffold = scaffold


def detect_multi_entity(query: str) -> "MultiEntityQuery | None":
    """Detect a comparison or intersection query over two or more entities.

    Returns a :class:`MultiEntityQuery` when the query both names at least two
    entities AND carries a comparison or intersection cue, else None. Single-entity
    queries (the default path) return None so the caller's normal retrieval runs."""
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
