"""Unit tests for natural-language category extraction (dealtype/industry
facets) and retrieval-query cleanup. The extractors resolve synonyms against the
*live* facet vocabulary stored in module globals, so tests install a fixture
vocabulary that mirrors the real index (e.g. funding -> 'Venture Capital',
not a mythical 'Funding' facet)."""

from app import main

# Mirror the real Qdrant facet labels so the synonym maps resolve as in prod.
_DEAL = ["Venture Capital", "M&A", "Private Equity", "Credit", "Investment Banking", "Markets"]
_IND = ["Finance", "Healthcare", "Education", "Technology", "Retail", "Cleantech"]
_CT = ["Article", "Interview", "Video", "Founder", "Competitor", "Appointment"]


def _set_facets() -> None:
    main._DEALTYPE_FACETS.clear()
    main._DEALTYPE_FACETS.update({d.lower(): d for d in _DEAL})
    main._INDUSTRY_FACETS.clear()
    main._INDUSTRY_FACETS.update({i.lower(): i for i in _IND})
    main._CONTENT_TYPE_FACETS.clear()
    main._CONTENT_TYPE_FACETS.update({c.lower(): c for c in _CT})


_ORIG_DEAL: dict = {}
_ORIG_IND: dict = {}
_ORIG_CT: dict = {}


def setup_function(_) -> None:
    global _ORIG_DEAL, _ORIG_IND, _ORIG_CT
    # Snapshot whatever the module globals currently hold so teardown can restore
    # them exactly, preventing any cross-module leakage (the extractors read these
    # module-global facet maps).
    _ORIG_DEAL = dict(main._DEALTYPE_FACETS)
    _ORIG_IND = dict(main._INDUSTRY_FACETS)
    _ORIG_CT = dict(main._CONTENT_TYPE_FACETS)
    _set_facets()


def teardown_function(_) -> None:
    # Restore the globals to their pre-test state rather than just clearing, so
    # other test modules never see this module's simulated vocabulary (nor lose
    # any state they had installed).
    main._DEALTYPE_FACETS.clear()
    main._DEALTYPE_FACETS.update(_ORIG_DEAL)
    main._INDUSTRY_FACETS.clear()
    main._INDUSTRY_FACETS.update(_ORIG_IND)
    main._CONTENT_TYPE_FACETS.clear()
    main._CONTENT_TYPE_FACETS.update(_ORIG_CT)


def test_extract_dealtype_funding() -> None:
    assert main.extract_dealtype("funding news") == "Venture Capital"
    assert main.extract_dealtype("fintech funding") == "Venture Capital"
    assert main.extract_dealtype("startup raised capital") == "Venture Capital"


def test_extract_dealtype_ma() -> None:
    assert main.extract_dealtype("biggest mergers") == "M&A"
    assert main.extract_dealtype("acquisitions news") == "M&A"
    assert main.extract_dealtype("buyout of the startup") == "M&A"


def test_extract_dealtype_private_equity() -> None:
    assert main.extract_dealtype("private equity deal") == "Private Equity"


def test_extract_dealtype_none_when_no_match() -> None:
    # No IPO facet exists in the corpus; extraction degrades to None (the query is
    # still cleaned so the embedding focuses on 'ipo').
    assert main.extract_dealtype("ipo news") is None
    assert main.extract_dealtype("latest news") is None
    assert main.extract_dealtype("how are you") is None


def test_extract_industry() -> None:
    assert main.extract_industry("fintech funding") == "Finance"
    assert main.extract_industry("healthtech deals") == "Healthcare"
    assert main.extract_industry("ecommerce funding") == "Retail"
    assert main.extract_industry("saas companies") == "Technology"
    assert main.extract_industry("cleantech startup") == "Cleantech"


def test_extract_industry_none() -> None:
    assert main.extract_industry("latest news") is None


def test_extract_content_type() -> None:
    # Content-type modifiers resolve to a real `content_type` facet value.
    assert main.extract_content_type("interviews with Narayana Murthy") == "Interview"
    assert main.extract_content_type("founders of Curefit") == "Founder"
    assert main.extract_content_type("competitors of Zomato") == "Competitor"
    assert main.extract_content_type("appointments in the TCS board") == "Appointment"
    assert main.extract_content_type("a video on fundraising") == "Video"


def test_extract_content_type_none_when_no_match() -> None:
    # Queries without a content-type modifier degrade to None (current behavior).
    assert main.extract_content_type("top funding deals 2025") is None
    assert main.extract_content_type("latest news") is None


def test_extract_content_type_unknown_facet_degrades() -> None:
    # When the corpus has no matching `content_type` facet, resolution degrades
    # to None rather than emitting a bogus value.
    main._CONTENT_TYPE_FACETS.clear()
    main._CONTENT_TYPE_FACETS.update({"article": "Article", "interview": "Interview"})
    assert main.extract_content_type("interviews with X") == "Interview"
    assert main.extract_content_type("founders of Y") is None
    assert main.extract_content_type("competitors of Z") is None


def test_build_facet_filter_content_type() -> None:
    # A content_type facet value becomes a Qdrant `content_type` filter condition.
    f = main.build_facet_filter(None, None, None, None, None, "Interview")
    assert f is not None
    assert [c.key for c in f.must] == ["content_type"]
    # None content_type yields no filter (current behavior).
    assert main.build_facet_filter(None, None, None, None, None, None) is None


def test_effective_intent_funding_news_june() -> None:
    rq, f, t, dt, ind = main._effective_intent("funding news in jun", None, None)
    # Date word ('jun') stripped; natural phrasing kept so rerank stays strong.
    assert rq == "funding news"
    assert dt == "Venture Capital"
    assert ind is None
    # June auto date filter is derived from the query.
    assert f is not None and t is not None
    assert f.startswith("20") and "-06-" in f


def test_effective_intent_ipo_relies_on_embedding() -> None:
    rq, _, _, dt, ind = main._effective_intent("ipo news", None, None)
    # No IPO facet exists, so the full phrase (embedding carries 'ipo') is kept.
    assert rq == "ipo news"
    assert dt is None and ind is None


def test_effective_intent_fintech_funding_year() -> None:
    _, f, t, dt, ind = main._effective_intent("top fintech funding 2025", None, None)
    assert dt == "Venture Capital"
    assert ind == "Finance"
    assert f == "2025-01-01" and t == "2025-12-31"


def test_effective_intent_no_facet() -> None:
    _, _, _, dt, ind = main._effective_intent("who is the CEO", None, None)
    assert dt is None and ind is None
