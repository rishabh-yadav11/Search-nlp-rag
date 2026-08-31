"""Tests for the recommendation engine (app/recommender.py).

These are unit tests that test the pure logic functions without requiring
Qdrant or Redis to be running. Integration tests that require the full stack
should be added separately.
"""
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCalculateRecencyScore:
    """Tests for _calculate_recency_score helper."""

    def test_recent_article(self):
        from app.recommender import _calculate_recency_score
        now = datetime.now(UTC)
        published = now.isoformat()
        score = _calculate_recency_score(published, now)
        assert 0.9 < score <= 1.0

    def test_old_article(self):
        from app.recommender import _calculate_recency_score
        now = datetime.now(UTC)
        old_date = (now - timedelta(days=365)).isoformat()
        score = _calculate_recency_score(old_date, now)
        assert score < 0.5

    def test_missing_date(self):
        from app.recommender import _calculate_recency_score
        now = datetime.now(UTC)
        score = _calculate_recency_score("", now)
        assert score == 0.5

    def test_invalid_date(self):
        from app.recommender import _calculate_recency_score
        now = datetime.now(UTC)
        score = _calculate_recency_score("not-a-date", now)
        assert score == 0.5

    def test_half_life_decay(self):
        from app.recommender import _calculate_recency_score
        now = datetime.now(UTC)
        # At 30 days, score = exp(-30/30) = 1/e ≈ 0.368
        thirty_days_ago = (now - timedelta(days=30)).isoformat()
        score = _calculate_recency_score(thirty_days_ago, now)
        assert abs(score - math.exp(-1)) < 0.01


class TestFormatArticles:
    """Tests for _format_articles helper."""

    def test_empty_points(self):
        from app.recommender import _format_articles
        result = _format_articles([])
        assert result == []

    def test_formats_valid_point(self):
        from app.recommender import _format_articles
        point = MagicMock()
        point.id = 42
        point.score = 0.85
        point.payload = {
            "title": "Test Article",
            "url": "https://example.com",
            "published_date": "2024-01-01T00:00:00",
            "category": "Tech",
            "summary": "A test summary",
            "author_names": ["Author One"],
            "industry_names": ["Technology"],
            "dealtype_names": ["Series A"],
        }
        result = _format_articles([point])
        assert len(result) == 1
        assert result[0]["id"] == 42
        assert result[0]["title"] == "Test Article"
        assert result[0]["score"] == 0.85

    def test_excludes_null_payload(self):
        from app.recommender import _format_articles
        point = MagicMock()
        point.id = 1
        point.score = 0.5
        point.payload = None
        result = _format_articles([point])
        assert result == []

    def test_excludes_specified_ids(self):
        from app.recommender import _format_articles
        point = MagicMock()
        point.id = 42
        point.score = 0.85
        point.payload = {
            "title": "Test Article",
            "url": "https://example.com",
        }
        result = _format_articles([point], exclude_ids=[42])
        assert result == []

    def test_missing_fields_have_defaults(self):
        from app.recommender import _format_articles
        point = MagicMock()
        point.id = 1
        point.score = 0.5
        point.payload = {"title": "Minimal", "url": "http://x"}
        result = _format_articles([point])
        assert result[0]["published_date"] is None
        assert result[0]["category"] is None
        assert result[0]["summary"] == ""
        assert result[0]["author_names"] == []


class TestRecommenderConfig:
    """Tests that config values are respected."""

    def test_enabled_by_default(self):
        from app.config import config
        assert config.ENABLE_RECOMMENDATIONS is True

    def test_default_weights(self):
        from app.config import config
        assert config.RECOMMEND_SIMILARITY_WEIGHT == 0.4
        assert config.RECOMMEND_CATEGORY_WEIGHT == 0.3
        assert config.RECOMMEND_RECENCY_WEIGHT == 0.2
        assert config.RECOMMEND_POPULARITY_WEIGHT == 0.1

    def test_disable_recommendations(self):
        from app.config import Config
        # Simulate disabled via env var
        original = Config.ENABLE_RECOMMENDATIONS
        try:
            with patch.dict('os.environ', {'ENABLE_RECOMMENDATIONS': 'false'}):
                # Re-import to pick up new env
                import importlib
                import app.config as config_module
                importlib.reload(config_module)
                assert config_module.config.ENABLE_RECOMMENDATIONS is False
        finally:
            Config.ENABLE_RECOMMENDATIONS = original


class TestGetSimilarArticlesDisabled:
    """Test similar articles when recommendations are disabled."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        from app.recommender import get_similar_articles
        with patch('app.recommender.config') as mock_config:
            mock_config.ENABLE_RECOMMENDATIONS = False
            result = await get_similar_articles(article_id=1)
            assert result == []


class TestGetLatestTopStoriesDisabled:
    """Test latest stories when recommendations are disabled."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        from app.recommender import get_latest_top_stories
        with patch('app.recommender.config') as mock_config:
            mock_config.ENABLE_RECOMMENDATIONS = False
            result = await get_latest_top_stories(limit=5)
            assert result == []


class TestUserProfileIntegration:
    """Integration tests for user profile interactions with Redis."""

    @pytest.mark.asyncio
    async def test_record_interaction_returns_none(self):
        """Test that recording an interaction returns None (success)."""
        from app.user_profile import record_interaction
        with patch('app.user_profile._redis_client') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            result = await record_interaction("user1", 42)
            assert result is None
            mock_client.pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_interactions_returns_empty_on_error(self):
        """Test graceful degradation when Redis is unavailable."""
        from app.user_profile import get_user_interactions
        with patch('app.user_profile._redis_client') as mock_redis:
            mock_redis.side_effect = Exception("Redis down")
            result = await get_user_interactions("user1")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_trending_articles_returns_empty_on_error(self):
        """Test graceful degradation for trending."""
        from app.user_profile import get_trending_articles
        with patch('app.user_profile._redis_client') as mock_redis:
            mock_redis.side_effect = Exception("Redis down")
            result = await get_trending_articles()
            assert result == []

    @pytest.mark.asyncio
    async def test_invalidate_user_profile_returns_none(self):
        """Test that invalidating profile works."""
        from app.user_profile import invalidate_user_profile
        with patch('app.user_profile._redis_client') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            result = await invalidate_user_profile("user1")
            assert result is None
