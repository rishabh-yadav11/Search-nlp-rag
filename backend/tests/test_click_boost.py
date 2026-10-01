"""Click boost: short-circuits, no-signal pass-through (incl. the Redis-down
degraded path), min-article-clicks + min-share gating, multiply + re-sort.
click_signals is faked on the module so no Redis client is needed."""

from types import SimpleNamespace

import pytest
from _support import run_sync as _run

from app import click_boost
from app.config import config


def _res(id_, score):
    return SimpleNamespace(id=id_, score=score)


def _no_signals():
    async def _f(_q):
        return None

    return _f


def _signals(by_id, total=None):
    async def _f(_q):
        return {"total": total if total is not None else sum(by_id.values()), "by_id": by_id}

    return _f


def _enable(monkeypatch, min_article=3, min_share=0.3, mult=1.3):
    monkeypatch.setattr(config, "ENABLE_CLICK_BOOST", True)
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_ARTICLE_CLICKS", min_article)
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_SHARE", min_share)
    monkeypatch.setattr(config, "CLICK_BOOST_MULT", mult)


# --- short-circuits ---


def test_disabled_short_circuit_skips_signals(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CLICK_BOOST", False)
    called = []

    async def _f(q):
        called.append(q)

    monkeypatch.setattr(click_boost, "click_signals", _f)
    results = [_res(1, 0.5)]

    out = _run(click_boost.apply_click_boost("q", results))

    assert out is results
    assert called == []


def test_empty_results_short_circuit_skips_signals(monkeypatch):
    _enable(monkeypatch)
    called = []

    async def _f(q):
        called.append(q)

    monkeypatch.setattr(click_boost, "click_signals", _f)

    assert _run(click_boost.apply_click_boost("q", [])) == []
    assert called == []


# --- no click signals ---


def test_no_click_signals_passthrough_unchanged(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(click_boost, "click_signals", _no_signals())
    results = [_res(1, 0.5), _res(2, 0.9)]

    out = _run(click_boost.apply_click_boost("q", results))

    assert out is results
    assert [r.score for r in out] == [0.5, 0.9]


def test_redis_down_degraded_signals_passthrough(monkeypatch):
    # ERROR PATH: Redis down makes click_signals return None, a silent pass-through.
    _enable(monkeypatch)
    monkeypatch.setattr(click_boost, "click_signals", _no_signals())
    results = [_res(1, 0.4), _res(2, 0.9)]

    out = _run(click_boost.apply_click_boost("q", results))

    assert [r.id for r in out] == [1, 2]
    assert [r.score for r in out] == [0.4, 0.9]


# --- boost loop: gating + multiply + re-sort ---


def test_boost_applies_gating_multiplies_and_resorts(monkeypatch):
    _enable(monkeypatch, min_article=3, min_share=0.3, mult=1.5)
    # total = 10 -> min_share = max(1, int(10*0.3)) = 3
    monkeypatch.setattr(click_boost, "click_signals", _signals({1: 5, 2: 2, 3: 8}))

    results = [_res(1, 0.5), _res(2, 0.9), _res(3, 0.8)]
    out = _run(click_boost.apply_click_boost("q", results))

    # Ids 1 and 3 clear MIN_ARTICLE_CLICKS; id 2 does not.
    assert {r.id: r.score for r in out} == {
        1: pytest.approx(0.75),
        2: 0.9,
        3: pytest.approx(1.2),
    }
    assert [r.id for r in out] == [3, 2, 1]


def test_boost_min_share_gate(monkeypatch):
    _enable(monkeypatch, min_article=1, min_share=0.5, mult=1.3)
    # total = 10 -> min_share = max(1, int(10*0.5)) = 5
    monkeypatch.setattr(click_boost, "click_signals", _signals({1: 4, 2: 6}))

    results = [_res(1, 0.5), _res(2, 0.5)]
    out = _run(click_boost.apply_click_boost("q", results))

    assert [r.id for r in out] == [2, 1]
    assert out[0].score == 0.5 * 1.3
    assert out[1].score == 0.5


def test_boost_min_share_floor_at_one(monkeypatch):
    _enable(monkeypatch, min_article=1, min_share=0.3, mult=2.0)
    # max(1, int(1*0.3)) = 1, so a single click qualifies.
    monkeypatch.setattr(click_boost, "click_signals", _signals({1: 1}))

    out = _run(click_boost.apply_click_boost("q", [_res(1, 0.5)]))

    assert out[0].score == 1.0


def test_boost_article_without_id_not_boosted(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(click_boost, "click_signals", _signals({1: 8}))

    no_id = SimpleNamespace(score=0.5)
    out = _run(click_boost.apply_click_boost("q", [no_id]))

    assert out[0].score == 0.5


def test_no_boost_change_keeps_original_order(monkeypatch):
    _enable(monkeypatch, min_article=100)
    monkeypatch.setattr(click_boost, "click_signals", _signals({1: 2}))
    results = [_res(1, 0.4), _res(2, 0.9)]

    out = _run(click_boost.apply_click_boost("q", results))

    # Nothing qualified -> no re-sort, so the lower score stays first.
    assert [r.id for r in out] == [1, 2]