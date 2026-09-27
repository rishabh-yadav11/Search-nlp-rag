"""Tests for the recommendation engine (app/recommender.py).

These are unit tests that test the pure logic functions without requiring
Qdrant or Redis to be running. Integration tests that require the full stack
should be added separately.
"""
import json
import logging
import math
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import ScoredPoint


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
            pipe = MagicMock()
            pipe.execute = AsyncMock()
            mock_client = AsyncMock()
            mock_client.pipeline = MagicMock(return_value=pipe)
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

    @pytest.mark.asyncio
    async def test_profile_vector_reads_json_string_as_float_list(self):
        """Consumers receive the ordered numeric vector stored in Redis."""
        from app.user_profile import get_user_profile_vector

        with patch("app.user_profile._redis_client") as mock_redis:
            client = AsyncMock()
            client.get.return_value = "[1, 2.5, 3]"
            mock_redis.return_value = client

            assert await get_user_profile_vector("user1") == [1.0, 2.5, 3.0]
            client.get.assert_awaited_once_with("user:profile_vector:user1")

    @pytest.mark.asyncio
    async def test_profile_vector_derives_and_persists_recent_interactions(self):
        """A cache miss derives a vector and category affinities from articles."""
        from app.user_profile import get_user_profile_vector

        now = datetime.now(UTC).timestamp()
        article = MagicMock(
            id=42,
            vector={"dense": [2.0, 4.0]},
            payload={"industry_names": ["technology"], "dealtype_names": ["merger"]},
        )
        qdrant = MagicMock()
        qdrant.retrieve = AsyncMock(return_value=[article])
        pipe = MagicMock()
        pipe.execute = AsyncMock()
        client = AsyncMock()
        client.get.return_value = None
        client.pipeline = MagicMock(return_value=pipe)

        with (
            patch("app.user_profile._redis_client", return_value=client),
            patch("app.user_profile.get_user_interactions", AsyncMock(return_value=[(42, now)])),
            patch.dict(sys.modules, {"app.main": MagicMock(state={"qdrant": qdrant})}),
        ):
            assert await get_user_profile_vector("user1") == [2.0, 4.0]

        qdrant.retrieve.assert_awaited_once()
        pipe.set.assert_called_once()
        assert json.loads(pipe.set.call_args.args[1]) == [2.0, 4.0]
        pipe.zadd.assert_called_once_with("user:categories:user1", {"technology": pytest.approx(1.0), "merger": pytest.approx(1.0)})


def _scored_point(pid: int, title: str) -> ScoredPoint:
    """A minimal Qdrant point the recommender can score and format."""
    return ScoredPoint(
        id=pid,
        version=0,
        score=0.5,
        payload={
            "title": title,
            "url": f"https://example.com/{pid}",
            "published_date": "2026-01-01T00:00:00Z",
        },
        vector=None,
    )


class _FakeQdrant:
    """Qdrant double whose vector, category and trending legs fail on demand.

    The vector leg queries with a point id (`query=...`); the category leg
    queries with only a filter. That is what lets a single fake fail one leg
    while the others keep working.
    """

    OUTAGE = "qdrant unreachable"

    def __init__(self, *, fail_vector=False, fail_category=False, fail_trending=False, fail_queries=()):
        self.fail_vector = fail_vector
        self.fail_category = fail_category
        self.fail_trending = fail_trending
        self.fail_queries = set(fail_queries)

    async def query_points(self, **kwargs):
        if "query" in kwargs:
            if self.fail_vector or kwargs["query"] in self.fail_queries:
                raise RuntimeError(self.OUTAGE)
            return SimpleNamespace(points=[_scored_point(11, "vector hit")])
        if self.fail_category:
            raise RuntimeError(self.OUTAGE)
        return SimpleNamespace(points=[_scored_point(12, "category hit")])

    async def scroll(self, **kwargs):
        if self.fail_trending:
            raise RuntimeError(self.OUTAGE)
        return ([_scored_point(13, "trending hit")], None)


async def _personalized(qdrant, *, interactions=(901,)):
    """Drive get_personalized_recommendations with a warm user profile."""
    from app import recommender

    now = datetime.now(UTC).timestamp()
    with (
        patch.object(recommender, "state", {"qdrant": qdrant}),
        patch.object(
            recommender, "get_user_interactions",
            AsyncMock(return_value=[(pid, now) for pid in interactions]),
        ),
        # A category containing "industry" is required for a category filter
        # to be built at all, otherwise the category leg short-circuits.
        patch.object(
            recommender, "get_user_profile_categories",
            AsyncMock(return_value=[("software industry", 1.0)]),
        ),
        patch.object(
            recommender, "get_trending_articles",
            AsyncMock(return_value=[{"article_id": 13}]),
        ),
    ):
        return await recommender.get_personalized_recommendations("user1", limit=5)


def _leg_warnings(caplog, leg, exc_message):
    """Warnings from ONE named leg that carry the exception text.

    Matching on the leg name matters: a warning from any other leg would
    otherwise satisfy the assertion, since they share the same exception.
    """
    return [
        record for record in caplog.records
        if record.name == "app.recommender"
        and record.levelno == logging.WARNING
        and leg in record.getMessage()
        and exc_message in record.getMessage()
    ]


_VECTOR_LEG = "Error getting vector candidates"
_CATEGORY_LEG = "Error getting category candidates"
_TRENDING_LEG = "Error getting trending candidates"


def _titles(result):
    return {article["title"] for article in result}


class TestCandidateLegObservability:
    """Each candidate leg must be visible when its data source fails."""

    def test_recommender_warnings_reach_caplog(self, caplog):
        """Guard: caplog is only meaningful if WARNING propagates to root."""
        logger = logging.getLogger("app.recommender")
        assert logger.propagate is True
        assert logger.getEffectiveLevel() <= logging.WARNING
        with caplog.at_level(logging.WARNING):
            logger.warning("probe warning")
        assert "probe warning" in caplog.text

    @pytest.mark.asyncio
    async def test_vector_leg_failure_logs_warning_with_exception(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = await _personalized(_FakeQdrant(fail_vector=True))

        assert _leg_warnings(caplog, _VECTOR_LEG, _FakeQdrant.OUTAGE), caplog.records
        # The other two legs still supply the feed.
        assert _titles(result) == {"category hit", "trending hit"}

    @pytest.mark.asyncio
    async def test_category_leg_failure_logs_warning_with_exception(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = await _personalized(_FakeQdrant(fail_category=True))

        assert _leg_warnings(caplog, _CATEGORY_LEG, _FakeQdrant.OUTAGE), caplog.records
        assert _titles(result) == {"vector hit", "trending hit"}

    @pytest.mark.asyncio
    async def test_trending_leg_failure_logs_warning_with_exception(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = await _personalized(_FakeQdrant(fail_trending=True))

        assert _leg_warnings(caplog, _TRENDING_LEG, _FakeQdrant.OUTAGE), caplog.records
        assert _titles(result) == {"vector hit", "category hit"}

    @pytest.mark.asyncio
    async def test_total_outage_logs_every_leg_once(self, caplog):
        """All three legs down: empty feed, and each leg warns exactly once.

        Once per leg because this user has a single interaction, and the
        vector handler logs per failing interaction rather than per request.
        """
        qdrant = _FakeQdrant(fail_vector=True, fail_category=True, fail_trending=True)
        with caplog.at_level(logging.WARNING):
            result = await _personalized(qdrant)

        assert result == []
        for leg in (_VECTOR_LEG, _CATEGORY_LEG, _TRENDING_LEG):
            assert len(_leg_warnings(caplog, leg, _FakeQdrant.OUTAGE)) == 1, (leg, caplog.records)

    @pytest.mark.asyncio
    async def test_vector_leg_keeps_results_from_interactions_that_worked(self, caplog):
        """One failed interaction is skipped; the rest of the leg still returns."""
        qdrant = _FakeQdrant(fail_queries=(901,))
        with caplog.at_level(logging.WARNING):
            result = await _personalized(qdrant, interactions=(901, 902))

        # The surviving interaction still contributed, alongside the other legs.
        assert _titles(result) == {"vector hit", "category hit", "trending hit"}
        # Exactly one vector warning, naming only the interaction that failed.
        vector_warnings = _leg_warnings(caplog, _VECTOR_LEG, _FakeQdrant.OUTAGE)
        assert len(vector_warnings) == 1, caplog.records
        assert "article 901" in vector_warnings[0].getMessage()
        assert "article 902" not in vector_warnings[0].getMessage()
