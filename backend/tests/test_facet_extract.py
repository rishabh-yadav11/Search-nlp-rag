"""Unit tests for natural-language category extraction (dealtype/industry
facets). The extractors resolve synonyms against the *live* facet vocabulary
stored in module globals, so tests install a fixture vocabulary first."""

from app import main

_DEAL = ["Funding", "IPO", "M&A", "Venture Debt", "Series A", "Stake Sale"]
_IND = ["Fintech", "Healthtech", "E-commerce", "SaaS", "Edtech"]


def _set_facets() -> None:
    main._DEALTYPE_FACETS.clear()
    main._DEALTYPE_FACETS.update({d.lower(): d for d in _DEAL})
    main._INDUSTRY_FACETS.clear()
    main._INDUSTRY_FACETS.update({i.lower(): i for i in _IND})


def setup_function(_) -> None:
    _set_facets()


def teardown_function(_) -> None:
    # The extractors read module-global facet maps; restore them so other test
    # modules (which don't simulate a loaded vocabulary) aren't polluted.
    main._DEALTYPE_FACETS.clear()
    main._INDUSTRY_FACETS.clear()


def test_extract_dealtype_ipo() -> None:
    assert main.extract_dealtype("ipo news") == "IPO"
    assert main.extract_dealtype("show me the latest IPO deals") == "IPO"
    assert main.extract_dealtype("initial public offering 2025") == "IPO"
    assert main.extract_dealtype("which companies listed this year") == "IPO"
    assert main.extract_dealtype("how many IPOs this year") == "IPO"


def test_extract_dealtype_funding() -> None:
    assert main.extract_dealtype("funding news") == "Funding"
    assert main.extract_dealtype("fintech funding") == "Funding"
    assert main.extract_dealtype("startup raised capital") == "Funding"


def test_extract_dealtype_ma() -> None:
    assert main.extract_dealtype("biggest mergers") == "M&A"
    assert main.extract_dealtype("acquisitions news") == "M&A"
    assert main.extract_dealtype("buyout of the startup") == "M&A"


def test_extract_dealtype_series() -> None:
    assert main.extract_dealtype("series a funding") == "Series A"


def test_extract_dealtype_none_when_no_match() -> None:
    assert main.extract_dealtype("latest news") is None
    assert main.extract_dealtype("how are you") is None


def test_extract_industry() -> None:
    assert main.extract_industry("fintech funding") == "Fintech"
    assert main.extract_industry("healthtech deals") == "Healthtech"
    assert main.extract_industry("ecommerce funding") == "E-commerce"
    assert main.extract_industry("healthcare startup") == "Healthtech"
    assert main.extract_industry("saas companies") == "SaaS"


def test_extract_industry_none() -> None:
    assert main.extract_industry("latest news") is None


def test_effective_intent_funding_news_june() -> None:
    _, f, t, dt, ind = main._effective_intent("funding news in jun", None, None)
    assert dt == "Funding"
    assert ind is None
    # June auto date filter is derived from the query.
    assert f is not None and t is not None
    assert f.startswith("20") and "-06-" in f


def test_effective_intent_fintech_funding_year() -> None:
    _, f, t, dt, ind = main._effective_intent("top fintech funding 2025", None, None)
    assert dt == "Funding"
    assert ind == "Fintech"
    assert f == "2025-01-01" and t == "2025-12-31"


def test_effective_intent_no_facet() -> None:
    _, _, _, dt, ind = main._effective_intent("who is the CEO", None, None)
    assert dt is None and ind is None
