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



class TestUserProfileIntegration:
    """Integration tests for user profile interactions with Redis."""

    @pytest.mark.asyncio
    async def test_record_interaction_reports_recorded(self):
        """A successful record reports RECORDED, and the guards run first.

        Replaces a mock-echo test that asserted the function returned None and
        that a MagicMock's pipeline was called once. The contract is now an
        explicit InteractionResult, and an article_id must clear the index
        check before any pipeline exists -- so this drives the real code with a
        fake that would raise on an unlisted command, and asserts both the
        result and that the counter was actually written.
        """
        from app.user_profile import InteractionResult, record_interaction

        written: dict[str, str] = {}

        class _FakeRedis:
            async def get(self, key):
                return None

            async def set(self, key, value, ex=None):
                written[key] = value
                return True

            def pipeline(self):
                # redis-py's pipeline() is SYNC-returning; the commands are then
                # executed with await. A coroutine here would never be awaited.
                class _Pipe:
                    def zcard(self, key):
                        return 0

                    def zscore(self, key, member):
                        return None

                    def zadd(self, *a, **k):
                        pass

                    def expire(self, *a, **k):
                        pass

                    def hset(self, *a, **k):
                        pass

                    def hincrby(self, key, field, amount):
                        written[f"{key}:{field}"] = str(amount)

                    def zincrby(self, *a, **k):
                        # Advances the trending index (#261) in the same
                        # transaction; this double only asserts the counter.
                        pass

                    def delete(self, *a, **k):
                        pass

                    async def execute(self):
                        return [0, None]

                return _Pipe()

        class _Qdrant:
            async def retrieve(self, **kwargs):
                return [SimpleNamespace(id=42)]

        with (
            patch("app.user_profile._redis_client", return_value=_FakeRedis()),
            patch.dict("app.main.state", {"qdrant": _Qdrant()}),
        ):
            result = await record_interaction("user1", 42, "click")

        assert result is InteractionResult.RECORDED
        # The article counter was genuinely written, by the legal field name.
        assert written["article:interactions:42:click"] == "1"

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


def _point_at(pid: int, published: datetime) -> ScoredPoint:
    """A scored point published at ``published``."""
    return ScoredPoint(
        id=pid,
        version=0,
        score=0.5,
        payload={
            "title": f"article {pid}",
            "url": f"https://example.com/{pid}",
            "published_date": published.isoformat(),
        },
        vector=None,
    )


class _PoolQdrant:
    """Qdrant double that serves each leg a pool exactly as wide as it is asked for.

    The leg named by ``fresh_leg`` hands out ids from its own base range with an
    age that *shrinks* down the pool, so its deepest candidates are the
    freshest; the other leg is STALE_DAYS old. Disjoint id ranges per leg plus
    recency in the hybrid score mean the page is filled from whichever
    candidates the pool actually reached, which is what makes the pool width
    visible in the response instead of being an internal fetch detail.
    """

    VECTOR_BASE = 100
    CATEGORY_BASE = 200
    STALE_DAYS = 90

    def __init__(self, *, fresh_leg="vector"):
        self.now = datetime.now(UTC)
        self.fresh_leg = fresh_leg

    def _leg_points(self, base, width, fresh):
        return [
            _point_at(
                base + i,
                self.now - timedelta(days=width - i if fresh else self.STALE_DAYS),
            )
            for i in range(width)
        ]

    async def query_points(self, **kwargs):
        width = kwargs["limit"]
        if "query" in kwargs:  # vector leg
            points = self._leg_points(self.VECTOR_BASE, width, fresh=self.fresh_leg == "vector")
        else:  # category leg
            points = self._leg_points(self.CATEGORY_BASE, width, fresh=self.fresh_leg == "category")
        return SimpleNamespace(points=points)

    async def scroll(self, **kwargs):
        return ([], None)


class _ScrollQdrant:
    """Qdrant double for the cold-start path, which only calls ``scroll``.

    Like the vector leg above, the deepest rows of the scroll are the freshest,
    so scrolling deeper is the only way to reach a fresher article.
    """

    BASE = 400

    def __init__(self):
        self.now = datetime.now(UTC)
        self.scrolled = 0

    async def scroll(self, **kwargs):
        width = self.scrolled = kwargs["limit"]
        points = [
            _point_at(self.BASE + i, self.now - timedelta(days=width - i))
            for i in range(width)
        ]
        return (points, None)


async def _pooled_page(candidates_limit, *, limit=5, fresh_leg="vector"):
    """Personalized feed for a warm user, with the candidate pool pinned."""
    from app import recommender

    now = datetime.now(UTC).timestamp()
    with (
        patch.object(recommender, "state", {"qdrant": _PoolQdrant(fresh_leg=fresh_leg)}),
        patch.object(recommender.config, "RECOMMEND_CANDIDATES_LIMIT", candidates_limit),
        patch.object(
            recommender, "get_user_interactions",
            AsyncMock(return_value=[(901, now)]),
        ),
        # A category containing "industry" is required for a category filter
        # to be built at all, otherwise the category leg short-circuits.
        patch.object(
            recommender, "get_user_profile_categories",
            AsyncMock(return_value=[("software industry", 1.0)]),
        ),
        patch.object(recommender, "get_trending_articles", AsyncMock(return_value=[])),
    ):
        return await recommender.get_personalized_recommendations("user1", limit=limit)


def _ids(result) -> set[int]:
    return {article["id"] for article in result}


def _of_leg(ids, base):
    return {i for i in ids if base <= i < base + 100}


class TestCandidatePoolWidth:
    """RECOMMEND_CANDIDATES_LIMIT must decide how deep the candidate pool goes."""

    @pytest.mark.asyncio
    async def test_deeper_pool_replaces_stale_candidates_on_the_page(self):
        """The knob is not cosmetic: a deeper pool changes which articles ship.

        With a pool of 5 the freshest reachable vector candidate is 5 days old
        and the page also carries the 90-day-old category hits; with a pool of
        25 the page can reach candidates a day old and the stale ones are
        outranked. Same user, same request, different pool.
        """
        narrow = await _pooled_page(2)
        wide = await _pooled_page(25)

        # limit=5 asks for a page of 10, so both pages are full.
        assert len(narrow) == len(wide) == 10
        assert _ids(narrow) != _ids(wide)

        # A 5-wide pool reaches only 5 fresh vector candidates and fills the rest
        # of the page with stale category hits; a 25-wide pool fills every slot.
        assert len(_of_leg(_ids(narrow), _PoolQdrant.VECTOR_BASE)) == 5
        assert _of_leg(_ids(narrow), _PoolQdrant.CATEGORY_BASE)
        assert len(_of_leg(_ids(wide), _PoolQdrant.VECTOR_BASE)) == 10
        assert not _of_leg(_ids(wide), _PoolQdrant.CATEGORY_BASE), (
            "a 25-wide pool reaches vector candidates fresher than the category hits, "
            "so no stale article should reach the page"
        )

    @pytest.mark.asyncio
    async def test_page_stays_full_when_the_knob_is_below_the_requested_limit(self):
        """A pool narrower than the request is raised to the request, not honoured.

        limit=8 asks for 8 candidates per leg, so a knob of 2 must not starve
        the page of candidates to choose from: the fetch is clamped up to 8 and
        nothing deeper than that is ever requested.
        """
        result = await _pooled_page(2, limit=8)

        assert len(result) == 16, "a starved pool cannot fill a page of 2 * limit"
        assert max(_of_leg(_ids(result), _PoolQdrant.VECTOR_BASE)) == _PoolQdrant.VECTOR_BASE + 7
        assert max(_of_leg(_ids(result), _PoolQdrant.CATEGORY_BASE)) == _PoolQdrant.CATEGORY_BASE + 7

    @pytest.mark.asyncio
    async def test_category_leg_pool_width_reaches_the_page(self):
        """The category leg is wired too, not just the vector one.

        Here the category leg is the fresh one, so the slots it can fill are
        bounded by its pool. At limit=5 the narrowest reachable pool is 5 (the
        clamp raises a knob of 3 to the page size) and fills 5 of the 10
        slots; a 20-wide pool fills all ten.
        """
        narrow = await _pooled_page(3, fresh_leg="category")
        wide = await _pooled_page(20, fresh_leg="category")

        assert len(narrow) == len(wide) == 10
        assert len(_of_leg(_ids(narrow), _PoolQdrant.CATEGORY_BASE)) == 5
        assert _of_leg(_ids(narrow), _PoolQdrant.VECTOR_BASE), "the stale leg fills the rest"
        assert len(_of_leg(_ids(wide), _PoolQdrant.CATEGORY_BASE)) == 10

    @pytest.mark.asyncio
    async def test_cold_start_knob_deepens_the_scroll_below_the_3x_floor(self):
        """The cold-start scroll honours the knob, on top of its own 3x.

        The scroll is called with ``over=3`` because the page is ``limit * 2``,
        and the knob is a floor on that rather than a cap: at limit=5 the floor
        is 15, so a knob of 3 cannot shrink it, while a knob of 20 deepens the
        scroll to 20 rows and the page is drawn from fresher rows instead.
        """
        from app import recommender

        async def _top_stories(candidates_limit):
            with (
                patch.object(recommender, "state", {"qdrant": _ScrollQdrant()}),
                patch.object(recommender.config, "RECOMMEND_CANDIDATES_LIMIT", candidates_limit),
            ):
                return await recommender._get_latest_top_stories(5)

        floor = await _top_stories(3)
        deepened = await _top_stories(20)

        assert len(floor) == len(deepened) == 10
        assert _ids(floor) != _ids(deepened), "a deeper scroll must change which rows the page comes from"
        # The 15-row floor builds the page out to row 14; a 20-row scroll
        # reaches ten rows further down.
        assert max(_ids(floor)) - _ScrollQdrant.BASE == 14
        assert max(_ids(deepened)) - _ScrollQdrant.BASE == 19

    @pytest.mark.asyncio
    async def test_cold_start_pool_never_narrows_below_the_page_headroom(self):
        """A knob below 3x must not shrink the cold-start scroll.

        The API accepts limit up to 20 and this path returns ``limit * 2``
        articles after dropping already-seen ids, so a scroll capped at the
        knob (50) instead of floored at it would fetch fewer rows than it did
        before this knob existed, and a page that comes up short once enough
        ids are excluded.
        """
        from app import recommender

        qdrant = _ScrollQdrant()
        excluded = [qdrant.BASE + i for i in range(11)]
        with (
            patch.object(recommender, "state", {"qdrant": qdrant}),
            # The shipped default, which is below 3 * 20 = 60.
            patch.object(recommender.config, "RECOMMEND_CANDIDATES_LIMIT", 50),
        ):
            result = await recommender._get_latest_top_stories(20, excluded)

        assert qdrant.scrolled == 60, "the 3x headroom must survive a knob below it"
        assert len(result) == 40, "a page of 2 * limit must not come up short after exclusions"
