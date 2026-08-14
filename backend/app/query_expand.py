"""Deterministic, dependency-free query expansion for the VCCircle news corpus.

There is a vocabulary gap between how users phrase a search and how articles are
written: users ask about "startup layoffs" while the corpus says "job cuts" and
"downsizing", or ask "who acquired X" while the article is framed as "X
acquisition". expand_query() detects such concepts with plain substring matching
and appends a small, bounded set of synonym phrases so semantic retrieval also
matches the corpus's wording. Stdlib only, fully offline.
"""

CONCEPT_EXPANSIONS: dict[str, list[str]] = {
    "funding/raise/raising/raised/raises/fundraise/fundraising": [
        "fundraise",
        "fundraising",
        "raised",
        "raises",
        "capital",
        "investment",
    ],
    "layoffs/layoff/laying off": [
        "lays off",
        "job cuts",
        "downsizing",
        "workforce reduction",
        "retrenchment",
        "layoff",
        "hiring freeze",
        "restructuring",
    ],
    "acquired/acquire/acquisition": [
        "acquisition",
        "acquire",
        "acquires",
        "buyout",
        "takeover",
    ],
    "startup/startups/start-up": [
        "start-up",
        "start-ups",
        "startup",
        "ventures",
    ],
    "ipo": [
        "initial public offering",
        "public listing",
        "stock market debut",
        "listing",
        "offer for sale",
        "OFs",
        "draft red herring prospectus",
        "DRHP",
    ],
    "grey market/grey market premium": [
        "grey market",
        "grey market premium",
        "unlisted premium",
        "pre-ipo",
    ],
    "m&a": [
        "mergers and acquisitions",
        "merger",
        "acquisition",
        "buyer",
        "seller",
        "takeover",
    ],
    "stake sale/stake sell": [
        "stake sale",
        "stake sell",
        "stake disposal",
        "partial exit",
        "share sale",
    ],
    "secondary sale/secondary deal": [
        "secondary sale",
        "secondary deal",
        "secondary transaction",
        "secondary market",
    ],
    "debt financing/venture debt": [
        "venture debt",
        "debt funding",
        "debt round",
    ],
    "edtech": [
        "education technology",
        "educational technology",
        "online learning",
        "edtech",
    ],
    "fintech": [
        "financial technology",
        "digital payments",
        "payments",
        "fintech",
    ],
    "funding round/investment round": [
        "investment round",
        "venture round",
        "series a",
        "series b",
        "series c",
    ],
}

_MAX_EXTRA_TOKENS = 6

_TRIGGERS: list[tuple[list[str], list[str]]] = [
    (key.split("/"), expansions) for key, expansions in CONCEPT_EXPANSIONS.items()
]


def expand_query(q: str) -> str:
    """Return ``q`` with a bounded set of corpus-friendly synonym phrases appended.

    Detection is a plain substring test on the lowercased query. Only concepts
    that actually appear in the query contribute expansions, terms already present
    in the query are dropped, and the appended words never exceed
    ``_MAX_EXTRA_TOKENS`` tokens. The original query text is preserved unchanged
    at the front. Returns ``q`` untouched when nothing matches.
    """
    q_lower = q.lower()
    candidates: list[str] = []
    seen: set[str] = set()
    for triggers, expansions in _TRIGGERS:
        if not any(trigger in q_lower for trigger in triggers):
            continue
        for term in expansions:
            term_lower = term.lower()
            if term_lower in seen or term_lower in q_lower:
                continue
            seen.add(term_lower)
            candidates.append(term)
    if not candidates:
        return q

    appended: list[str] = []
    remaining = _MAX_EXTRA_TOKENS
    for term in candidates:
        tokens = len(term.split())
        if tokens > remaining:
            continue
        appended.append(term)
        remaining -= tokens
        if remaining == 0:
            break
    if not appended:
        return q
    return f"{q.rstrip()} {' '.join(appended)}"
