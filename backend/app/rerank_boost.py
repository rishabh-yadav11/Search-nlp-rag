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

# Common nouns / sector labels that are NOT proper-noun entities. A capitalized
# word in this set is never treated as a standalone entity, and it is stripped
# from the tail of a multi-word entity phrase. This stops a bare sector noun
# (e.g. "internet" / "consumer") from over-boosting unrelated articles, and
# keeps a query like "consumer internet" from drifting to every "internet" hit.
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

# Legal-entity suffixes stripped from the tail of a multi-word entity phrase so
# e.g. "Banyan Netfaqs Pvt Ltd" resolves to the distinct entity "banyan
# netfaqs" rather than the bare, over-broad token "banyan".
_ENTITY_SUFFIXES = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llp", "llc", "plc", "sa", "ag",
}

# A run of two or more consecutive capitalized words is treated as a single
# proper-noun phrase (a company / fund / person name spoken as one entity),
# rather than being exploded into individual tokens that would each boost
# independently and conflate distinct entities sharing a headword.
_RUN_RE = re.compile(r"[A-Z][A-Za-z0-9.']+(?:\s+[A-Z][A-Za-z0-9.']+)+")


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
    """Normalize a multi-word capitalized run into one entity: lowercased, with
    trailing legal-entity suffixes and generic tail nouns removed. Returns "" when
    the run is empty or consists solely of generic nouns."""
    words = [_normalize(w) for w in phrase.split()]
    while len(words) > 1 and (words[-1] in _GENERIC_NOUNS or words[-1] in _ENTITY_SUFFIXES):
        words.pop()
    if not words or all(w in _GENERIC_NOUNS for w in words):
        return ""
    return " ".join(words)


def extract_entities(q: str) -> list[str]:
    """Extract proper-noun-like entities from a query. Handles known brand names
    (including spaces and apostrophes) plus multi-word capitalized phrases,
    normalizes to lowercase with possessive apostrophes removed, and returns
    distinct entities.

    Consecutive capitalized words are kept as ONE entity (e.g. "Banyan Netfaqs
    Pvt Ltd" -> "banyan netfaqs"), so distinct entities sharing a headword are
    not conflated, and generic sector nouns ("consumer internet") are not
    over-expanded into bare tokens that over-boost unrelated articles."""
    nq = _normalize(q)
    if not nq:
        return []
    raw: list[str] = [m.group(0) for m in _BRAND_RE.finditer(nq)]
    for run in _RUN_RE.findall(q):
        phrase = _strip_entity_phrase(run)
        if phrase and phrase not in raw:
            raw.append(phrase)
    ordered: list[str] = []
    for e in raw:
        if e not in ordered:
            ordered.append(e)
    # Drop any entity fully contained in a longer one (e.g. "acme" inside
    # "acme corp"). Sort by descending length so the longer entity is kept first
    # and the shorter substring is rejected in a single pass. A known brand is
    # never dropped even when a longer non-brand run subsumes it (e.g. "ola
    # electric" must survive the run "ola electric ipo price band").
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
    """Return a NEW list of copies of `results` whose `.score` is boosted when a
    query entity appears in the result title (BOOST_TITLE) or, failing that, in
    the summary (BOOST_SUMMARY). Inputs are never mutated. Boosted scores are
    sorted descending; ties keep the original input order (stable sort)."""
    entities = extract_entities(q)
    if not entities:
        return list(results)
    # Compile once per entity: require a word boundary at the START (so "ola"
    # does NOT match inside "solar"/"polar"), allow a trailing non-letter (so
    # "tcs" still matches "tcs2024"/"tcs." but not "tcsql"), and tolerate a
    # possessive suffix ("'s" or bare "s") so "Ola Electric" still matches the
    # apostrophe-stripped "ola electrics" in a summary.
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
        if _matches(title):
            new_score = r.score * BOOST_TITLE
        elif _matches(summary):
            new_score = r.score * BOOST_SUMMARY
        else:
            new_score = r.score
        clone = copy.copy(r)
        clone.score = new_score
        boosted.append((new_score, clone))
    boosted.sort(key=lambda t: t[0], reverse=True)
    return [clone for _, clone in boosted]
