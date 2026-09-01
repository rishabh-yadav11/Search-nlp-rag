"""Validate the slim SourceSummary public DTO (parallel task).

The DTO must carry only the fields needed by the frontend plus a short `summary`
excerpt for editors, and must never leak the full article `body` in API responses.
"""

from app.main import SourceArticle, SourceSummary


def _sample_article() -> SourceArticle:
    return SourceArticle(
        id=42,
        title="A deal",
        url="https://example.com/42",
        published_date="2025-06-01T00:00:00+00:00",
        category="M&A",
        summary="Internal summary text",
        body="Full body text that must never be exposed",
        author_names=["Alice"],
        industry_names=["Fintech"],
        dealtype_names=["Series A"],
        score=0.9,
    )


def test_source_summary_includes_summary_but_never_body():
    article = _sample_article()
    summary = SourceSummary(
        id=article.id,
        title=article.title,
        url=article.url,
        published_date=article.published_date,
        category=article.category,
        summary=article.summary,
        score=article.score,
    )
    dumped = summary.model_dump()
    for field in ("id", "title", "url", "published_date", "category", "score", "summary"):
        assert field in dumped
    assert "body" not in dumped


def test_source_summary_dump_contains_only_public_fields():
    article = _sample_article()
    summary = SourceSummary.model_validate(
        {
            "id": article.id,
            "title": article.title,
            "url": article.url,
            "published_date": article.published_date,
            "category": article.category,
            "score": article.score,
            "author_names": article.author_names,
            "industry_names": article.industry_names,
            "dealtype_names": article.dealtype_names,
        }
    )
    public_fields = {
        "id",
        "title",
        "url",
        "published_date",
        "category",
        "score",
        "summary",
        "author_names",
        "industry_names",
        "dealtype_names",
        "content_type",
    }
    assert set(summary.model_dump()) == public_fields
    assert summary.summary == ""


def test_source_summary_serializable_from_article():
    article = _sample_article()
    summary = SourceSummary.model_validate(article.model_dump())
    assert summary.id == article.id
    assert summary.title == article.title
    assert summary.score == article.score
    assert summary.summary == article.summary
    assert summary.model_dump_json()
