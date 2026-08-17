"""The LLM module must resolve settings from the config *instance* (`app.config.config`),
not the `app.config` module object. Regression test for a runtime AttributeError that
the unit tests missed because they never exercised llm.generate_answer."""

import pytest

from app import llm
from app.config import config as config_instance
from app.llm import LLMResult


def test_llm_reads_config_instance_attributes():
    assert llm.config is config_instance
    assert isinstance(llm.config.LLM_TIMEOUT_SECONDS, (int, float))
    assert llm.config.LLM_MAX_RETRIES >= 0
    assert llm.config.LLM_RETRY_BACKOFF > 0


def test_gemini_config_present():
    assert hasattr(config_instance, "GEMINI_API_KEY")
    assert hasattr(config_instance, "GEMINI_BASE_URL")
    assert config_instance.GEMINI_BASE_URL.startswith("https://")
    assert config_instance.LLM_MODEL  # non-empty default model


def test_llm_result_cost_in_inr(monkeypatch):
    # 1M input @ $0.59 + 1M output @ $0.79 = $1.38; * 95.60 INR/USD = 131.928
    monkeypatch.setattr(config_instance, "LLM_PRICE_INPUT_PER_1M", 0.59)
    monkeypatch.setattr(config_instance, "LLM_PRICE_OUTPUT_PER_1M", 0.79)
    monkeypatch.setattr(config_instance, "INR_PER_USD", 95.60)
    r = LLMResult(content="x", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert r.cost() == pytest.approx(131.928, abs=1e-9)
    assert r.total_tokens == 2_000_000
