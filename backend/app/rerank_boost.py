import copy
import re

BOOST_TITLE = 1.25
BOOST_SUMMARY = 1.10

_BRAND_ENTITIES = [
    "PhonePe",
    "Paytm",
    "Flipkart",
    "Zomato",
    "Swiggy",
    "BYJU'S",
    "BYJUS",
    "Housing.com",
    "Housing",
    "PayU",
    "Pine Labs",
    "Ola Electric",
    "Razorpay",
    "Zepto",
    "Nykaa",
    "FirstCry",
    "Blinkit",
    "NPST",
    "NSDL",
    "NSE",
    "Prosus",
    "SoftBank",
    "Ant Group",
    "Temasek",
    "Blackstone",
    "KKR",
    "Sequoia",
    "Tiger Global",
    "Accel",
    "Meesho",
    "Mamaearth",
    "ShareChat",
    "Grofers",
    "BigBasket",
    "Myntra",
    "Dunzo",
    "Rapido",
    "Udaan",
    "CRED",
    "BharatPe",
    "Groww",
    "Zerodha",
    "PolicyBazaar",
    "PharmEasy",
    "Practo",
    "CureFit",
    "Unacademy",
    "Vedantu",
    "upGrad",
    "Ather Energy",
    "Ather",
    "Ola",
    "Uber",
    "Infosys",
    "TCS",
    "Wipro",
    "Reliance",
    "Tata",
    "Adani",
    "Lightspeed",
    "Elevation",
    "Blume",
    "Chiratae",
    "Kalaari",
    "Nexus",
    "Matrix Partners",
    "Khosla",
    "General Atlantic",
    "Coatue",
    "Warburg Pincus",
    "Warburg",
    "TPG",
    "GIC",
    "LGT",
    "Alteria",
    "Stride",
    "Peak XV",
    "InnoVen",
    "IntelleGrow",
    "Lendingkart",
    "MakeMyTrip",
    "Goibibo",
    "Oyo",
    "Freshworks",
    "Chargebee",
    "Zoho",
    "Navis",
    "Canada Pension Plan",
    "CPPIB",
    "AB InBev",
    "Maruti",
    "Mahindra",
    "Hero",
    "Bajaj",
    "Ashok Leyland",
]

# Capitalized words never treated as entities, and stripped from a phrase tail:
# a bare sector noun ("internet") would otherwise over-boost unrelated articles.
_GENERIC_NOUNS = {
    "consumer", "internet", "sector", "sectors", "industry", "industries",
    "market", "markets", "funding", "news", "deal", "deals", "company",
    "companies", "startup", "startups", "outlook", "growth", "latest",
    "business", "technology", "tech", "services", "service", "solution",
    "solutions", "digital", "online", "report", "reports", "update",
    "updates", "trend", "trends", "analysis", "view", "views", "story",
    "stories", "round", "rounds", "fund", "funds", "capital", "venture",
    "india", "indian", "global", "domestic", "foreign", "year", "years",
    "quarter", "month", "months",
}

# Legal-entity suffixes stripped from a phrase tail so "Banyan Netfaqs Pvt Ltd"
# resolves to "banyan netfaqs", not the over-broad bare token "banyan".
_ENTITY_SUFFIXES = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llp", "llc", "plc", "sa", "ag",
}

# Trailing query-context nouns stripped from a phrase tail so "Banyan Netfaqs IPO
# Price Band" resolves to "banyan netfaqs" and matches the company as other
# articles name it. Stripping stops at len==1, so a context word can never
# collapse a distinct multi-word entity into one bare token.
_CONTEXT_NOUNS = {
    "ipo", "price", "band", "bands", "result", "results", "earnings",
    "share", "shares", "stock", "stocks", "outlook", "performance",
    "profit", "loss", "revenue", "quarter", "q1", "q2", "q3", "q4",
    "fy", "fiscal", "financial", "subsidiary", "holdings", "group",
    "ventures", "capital", "partners", "enterprises", "industries",
}

# Two or more consecutive capitalized words are ONE entity, not tokens boosted
# separately, which would conflate distinct entities sharing a headword.
_RUN_RE = re.compile(r"[A-Z][A-Za-z0-9.']+(?:\s+[A-Z][A-Za-z0-9.']+)+")

# A capitalized word outside any multi-word run becomes its own entity, so
# single-word companies boost without being in the curated brand list; span
# checks in extract_entities exclude matches inside a longer run.
_SINGLE_CAP_RE = re.compile(r"\b[A-Z][A-Za-z0-9.']+\b")

# Never treated as standalone entities: question/function words capitalized at
# a query's start, plus the generic-noun and query-context sets.
_SINGLE_WORD_IGNORE = (
    _GENERIC_NOUNS
    | _CONTEXT_NOUNS
    | {
        "what", "why", "how", "when", "where", "which", "who", "whom",
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "my", "our", "your",
    }
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("'", "").replace("\u2019", "").strip())


_NORMALIZED_BRANDS = sorted(
    {_normalize(b) for b in _BRAND_ENTITIES},
    key=lambda b: (len(b), b),
    reverse=True,
)
_BRAND_SET = set(_NORMALIZED_BRANDS)
_BRAND_RE = re.compile(r"\b(?:" + "|".join(re.escape(b) for b in _NORMALIZED_BRANDS) + r")\b")


def _strip_entity_phrase(phrase: str) -> str:
    """Lowercased base entity with legal-entity suffixes, generic nouns and
    query-context nouns stripped; "" when nothing recognizable is left."""
    words = [_normalize(w) for w in phrase.split()]
    _strip_trail = _GENERIC_NOUNS | _ENTITY_SUFFIXES | _CONTEXT_NOUNS
    while len(words) > 1 and words[-1] in _strip_trail:
        words.pop()
    while len(words) > 1 and (words[0] in _GENERIC_NOUNS or words[0] in _ENTITY_SUFFIXES):
        words.pop(0)
    if not words or all(w in _strip_trail for w in words):
        return ""
    return " ".join(words)


def extract_entities(q: str) -> list[str]:
    """Extract distinct proper-noun-like entities from a query: known brands
    plus multi-word capitalized runs, lowercased and apostrophe-stripped."""
    nq = _normalize(q)
    if not nq:
        return []
    raw: list[str] = [m.group(0) for m in _BRAND_RE.finditer(nq)]
    run_spans = [m.span() for m in _RUN_RE.finditer(q)]
    for run in _RUN_RE.findall(q):
        phrase = _strip_entity_phrase(run)
        if phrase and phrase not in raw:
            raw.append(phrase)
    # Capitalized words outside a multi-word run and not in _SINGLE_WORD_IGNORE
    # become their own entity, so single-word companies boost too.
    for m in _SINGLE_CAP_RE.finditer(q):
        s, e = m.span()
        if any(lo <= s < hi for lo, hi in run_spans):
            continue
        word = _normalize(m.group(0))
        if word in _SINGLE_WORD_IGNORE:
            continue
        if word not in raw:
            raw.append(word)
    ordered: list[str] = []
    for e in raw:
        if e not in ordered:
            ordered.append(e)
    # Drop an entity fully contained in a longer kept one; descending-length
    # order decides the longer one first. A known brand is never dropped by a
    # longer non-brand run that subsumes it.
    ordered.sort(key=len, reverse=True)
    kept: list[str] = []
    for e in ordered:
        if e in _BRAND_SET:
            if e not in kept:
                kept.append(e)
            continue
        if not any(e != k and e in k for k in kept):
            kept.append(e)
    return kept


def apply_entity_boost(q: str, results: list) -> list:
    """Return NEW copies of `results` with `.score` boosted when a query entity
    appears in the title (BOOST_TITLE) else the summary (BOOST_SUMMARY)."""
    entities = extract_entities(q)
    if not entities:
        return list(results)
    # Word boundary at the START only ("ola" must not match "solar"), a trailing
    # non-letter allowed ("tcs2024" matches, "tcsql" does not), and a possessive
    # suffix tolerated so "Ola Electric" matches "ola electrics".
    entity_res = [
        re.compile(rf"\b{re.escape(e)}(?:['’]?s)?(?![a-zA-Z])", re.IGNORECASE)
        for e in entities
    ]

    def _matches(text: str) -> bool:
        return any(p.search(text) for p in entity_res)

    boosted = []
    for r in results:
        title = _normalize(r.title or "")
        summary = _normalize(getattr(r, "summary", "") or "")
        score = getattr(r, "score", None)
        if score is None:
            score = 0.0
        if _matches(title):
            new_score = score * BOOST_TITLE
        elif _matches(summary):
            new_score = score * BOOST_SUMMARY
        else:
            new_score = score
        clone = copy.copy(r)
        clone.score = new_score
        boosted.append((new_score, clone))
    boosted.sort(key=lambda t: t[0], reverse=True)
    return [clone for _, clone in boosted]
