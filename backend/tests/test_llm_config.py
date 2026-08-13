"""The LLM module must resolve settings from the config *instance* (`app.config.config`),
not the `app.config` module object. Regression test for a runtime AttributeError that
the unit tests missed because they never exercised llm.generate_answer."""

from app import llm
from app.config import config as config_instance


def test_llm_reads_config_instance_attributes():
    assert llm.config is config_instance
    assert isinstance(llm.config.LLM_TIMEOUT_SECONDS, (int, float))
    assert llm.config.LLM_MAX_RETRIES >= 0
    assert llm.config.LLM_RETRY_BACKOFF > 0
