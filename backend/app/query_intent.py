import calendar
import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

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


_MONTH_SPAN_RE = re.compile(
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?\s*(?:-|to|through|and)\s*"
    rf"\b({_MONTH_ALT})\b(?:\s+(?:of\s+)?((?:19|20)\d{{2}}))?",
    re.IGNORECASE,
)

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









_FY_RE = re.compile(
    r"\bfy\s*(?P<fy_range_start>(?:19|20)?\d{2})\s*(?:-|to|through)\s*(?P<fy_range_end>(?:19|20)?\d{2})\b"
    r"|\bfy\s*'?(?P<fy_single>(?:19|20)?\d{2})\b"
    r"|\bfiscal\s+year\s*(?P<fiscal_year>(?:19|20)?\d{2})\b"
    r"|\bfiscal\s*(?P<fiscal_bare>(?:19|20)?\d{2})\b",
    re.IGNORECASE,
)




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

    q = query.lower()




    matches = list(_FY_RE.finditer(q))
    span = next((m for m in matches if m.group("fy_range_start")), None)
    if span:
        y1 = _full_year(int(span.group("fy_range_start")), _current_year())
        y2 = _full_year(int(span.group("fy_range_end")), y1)





        start = min(y1, y2)
        end = max(y1, y2)



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






_RECENCY_WINDOW_RE = re.compile(
    r"\b(?:today|"
    r"this\s+week|past\s+week|last\s+week|"
    r"this\s+month|past\s+month|last\s+month|"
    r"(?:past|last)\s+(\d+)\s*(day|days|week|weeks|month|months)|"
    r"(\d+)\s*(day|days|week|weeks|month|months)\s*ago)\b",
    re.IGNORECASE,
)



_RECENCY_INTENT_RE = re.compile(
    r"\b(latest|recent|newest|freshest|fresh|lately|breaking|of\s+late)\b",
    re.IGNORECASE,
)

_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


def _days_ago_iso(days: int) -> str:

    return (_today() - timedelta(days=days)).isoformat()


def _month_start_iso() -> str:

    return _today().replace(day=1).isoformat()


def _week_start_iso() -> str:

    t = _today()
    return (t - timedelta(days=t.weekday())).isoformat()


def extract_recency_range(query: str) -> tuple[str, str] | None:

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

    return _RECENCY_WINDOW_RE.sub(" ", query).strip()


def strip_recency_intent(query: str) -> str:

    return _RECENCY_INTENT_RE.sub(" ", query).strip()


def is_recency_intent(query: str) -> bool:

    return bool(_RECENCY_INTENT_RE.search(query))


def extract_year_range(query: str) -> tuple[str, str] | None:

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

    def _replace(m: re.Match) -> str:
        n = _top_n_to_int(m.group(2))
        return f"{m.group(1)} {n}" if n is not None else m.group(0)

    return _TOP_N_RE.sub(_replace, query)


def suggested_top_k(query: str) -> int | None:

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

    if extract_month_range(query) is not None:
        return query, False
    topic = extract_list_topic(query)
    yr = _referenced_year(query)
    if yr is None or topic is None:
        return query, False
    new_q = f"Flashback {yr} {topic}".strip()
    return new_q, new_q != query


def _referenced_year(query: str) -> int | None:

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






_ACQUIRE_VERB_RE = re.compile(
    r"\b(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b", re.IGNORECASE
)


_ACTIVE_ACQ_PREDICATE = (
    r"(?:has|have|had)\s+(?:acquired|bought|purchased|taken\s*over)"
    r"|(?:is|are|was|were)\s+(?:acquiring|buying|taking\s*over|(?:the\s+)?acquirers?)"
    r"|acquired|acquires|bought|buys|purchased|purchases|took\s*over|takes?\s*over"
)




_WHO_ACQUIRED_RE = re.compile(
    rf"\b(?:who|which\s+(?:compan(?:y|ies)|firms?|business(?:es)?))\s+(?:{_ACTIVE_ACQ_PREDICATE})\b",
    re.IGNORECASE,
)










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

    if not _ACQUIRE_VERB_RE.search(query):
        return None



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

    s = _strip_entities(query.lower(), entities)
    s = _COMPARE_RE.sub(" ", s)
    s = _BOTH_AND_RE.sub(" ", s)
    s = _ALL_OF_RE.sub(" ", s)
    s = re.sub(r"\b(?:both|all|and|with|the|of|for|to|in|by|from|between)\b", " ", s)
    s = re.sub(r"[\s-]+", " ", s).strip()
    return s


class MultiEntityQuery:


    def __init__(self, mode: str, entities: list[str], scaffold: str):
        self.mode = mode
        self.entities = entities
        self.scaffold = scaffold


def detect_multi_entity(query: str) -> "MultiEntityQuery | None":

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
