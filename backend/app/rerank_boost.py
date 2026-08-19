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

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}

_CAP_RE = re.compile(r"[A-Z][a-zA-Z0-9.]*")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("'", "").replace("\u2019", "").strip())


_NORMALIZED_BRANDS = sorted(
    {_normalize(b) for b in _BRAND_ENTITIES},
    key=lambda b: (len(b), b),
    reverse=True,
)
_BRAND_RE = re.compile(r"\b(?:" + "|".join(re.escape(b) for b in _NORMALIZED_BRANDS) + r")\b")


def extract_entities(q: str) -> list[str]:
    """Extract proper-noun-like entities from a query. Handles known brand names
    (including spaces and apostrophes) plus capitalized words/phrases, strips
    stopwords, and returns entities normalized to lowercase with possessive
    apostrophes removed."""
    nq = _normalize(q)
    if not nq:
        return []
    raw = [m.group(0) for m in _BRAND_RE.finditer(nq)]
    raw.extend(_normalize(t) for t in _CAP_RE.findall(q))
    raw = [e for e in raw if e not in _STOPWORDS]
    ordered: list[str] = []
    for e in raw:
        if e not in ordered:
            ordered.append(e)
    return [e for e in ordered if not any(e != other and e in other for other in ordered)]


def apply_entity_boost(q: str, results: list) -> list:
    """Return a NEW list of copies of `results` whose `.score` is boosted when a
    query entity appears in the result title (BOOST_TITLE) or, failing that, in
    the summary (BOOST_SUMMARY). Inputs are never mutated. Boosted scores are
    sorted descending; ties keep the original input order (stable sort)."""
    entities = extract_entities(q)
    if not entities:
        return list(results)
    boosted = []
    for r in results:
        title = _normalize(r.title or "")
        summary = _normalize(getattr(r, "summary", "") or "")
        if any(e in title for e in entities):
            new_score = r.score * BOOST_TITLE
        elif any(e in summary for e in entities):
            new_score = r.score * BOOST_SUMMARY
        else:
            new_score = r.score
        clone = copy.copy(r)
        clone.score = new_score
        boosted.append((new_score, clone))
    boosted.sort(key=lambda t: t[0], reverse=True)
    return [clone for _, clone in boosted]
