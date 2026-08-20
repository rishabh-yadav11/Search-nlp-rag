from app import query_expand
from app.query_expand import expand_query


def test_expand_layoffs_query_with_job_cut_terms():
    expanded = expand_query("startup layoffs India 2025")
    assert expanded.startswith("startup layoffs India 2025")
    assert "job cuts" in expanded
    assert "downsizing" in expanded


def test_expand_acquired_query_with_acquisition_synonyms():
    expanded = expand_query("who acquired Housing.com")
    assert expanded.startswith("who acquired Housing.com")
    assert "acquisition" in expanded
    assert "buyout" in expanded


def test_expand_funding_query():
    expanded = expand_query("startups raising money")
    assert expanded.startswith("startups raising money")
    assert "fundraise" in expanded
    assert "investment" in expanded


def test_expand_ipo_query():
    expanded = expand_query("Ola Electric IPO")
    assert expanded.startswith("Ola Electric IPO")
    assert "initial public offering" in expanded
    assert "public listing" in expanded


def test_expand_edtech_query():
    expanded = expand_query("top edtech companies")
    assert "education technology" in expanded


def test_expand_no_concept_returns_unchanged():
    assert expand_query("top 10 companies in India") == "top 10 companies in India"
    assert expand_query("") == ""


def test_expand_is_bounded_to_six_extra_tokens():
    for q in ("startup layoffs India 2025", "who acquired Housing.com", "startups raising money"):
        original_tokens = len(q.split())
        expanded = expand_query(q)
        assert len(expanded.split()) - original_tokens <= 6


def test_ai_concept_expands_to_multi_token_terms_by_default():
    expanded = expand_query("ai")
    assert "artificial intelligence" in expanded
    assert "machine learning" in expanded


def test_expand_returns_query_unchanged_when_no_term_fits_token_budget(monkeypatch):
    monkeypatch.setattr(query_expand, "_MAX_EXTRA_TOKENS", 1)
    assert expand_query("ai") == "ai"
