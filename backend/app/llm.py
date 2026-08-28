import asyncio
import logging
import random
from dataclasses import dataclass

import openai

from app.config import config

logger = logging.getLogger("llm")


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be reached, after retries are exhausted."""

    def __init__(self, message: str = "LLM temporarily unavailable"):
        super().__init__(message)


@dataclass
class LLMResult:
    """LLM answer plus reported token usage (for cost tracking)."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self) -> float:
        """Estimated cost of this call in INR, from config pricing per 1M tokens
        (USD) converted at INR_PER_USD."""
        usd = (
            self.prompt_tokens / 1_000_000 * config.LLM_PRICE_INPUT_PER_1M
            + self.completion_tokens / 1_000_000 * config.LLM_PRICE_OUTPUT_PER_1M
        )
        return usd * config.INR_PER_USD


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        # Retry only transient failures: 5xx server errors and rate-limit (429).
        # Do NOT retry 4xx client errors (e.g. 400 invalid request) — they won't
        # succeed on retry and would just burn attempts/budget.
        return exc.status_code is not None and (exc.status_code >= 500 or exc.status_code == 429)
    return False


async def generate_answer(llm_client, prompt: str, model: str) -> LLMResult:
    """Call the LLM with a timeout and retries on transient errors.

    Retries exponential backoff (LLM_RETRY_BACKOFF * 2^attempt) up to
    LLM_MAX_RETRIES. Raises LLMUnavailableError (wrapping the last error)
    after retries are exhausted, so callers never surface raw SDK errors.
    Returns an LLMResult with the answer text and token usage.
    """
    last_error = None
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            usage = response.usage
            if not response.choices:
                raise LLMUnavailableError("LLM returned an empty choices list") from None
            return LLMResult(
                content=response.choices[0].message.content or "",
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            )
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt >= config.LLM_MAX_RETRIES:
                raise LLMUnavailableError() from exc
            delay = config.LLM_RETRY_BACKOFF * (2**attempt)
            delay += random.uniform(0, delay * 0.5)
            logger.warning(
                "LLM call failed on attempt %d/%d (%s); retrying in %.1fs",
                attempt + 1,
                config.LLM_MAX_RETRIES + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    raise LLMUnavailableError() from last_error


async def stream_answer(llm_client, prompt: str, model: str, usage_holder: list | None = None):
    """Yield answer text chunks as they arrive from the LLM.

    Same retry policy as generate_answer, but only retries when the stream
    fails before yielding any content (a mid-stream failure would otherwise
    duplicate already-sent text). Each item yielded is a string chunk; the
    caller reassembles the full answer. When usage_holder is provided (a
    single-element list), it is filled with the final LLMResult after the
    stream completes. Raises LLMUnavailableError after retries are exhausted.
    """
    last_error = None
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        started = False
        try:
            stream = await llm_client.chat.completions.create(
                model=model,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
                timeout=config.LLM_TIMEOUT_SECONDS,
                stream=True,
                stream_options={"include_usage": True},
            )
            usage = None
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = (delta.content or "") if delta else ""
                if piece:
                    started = True
                    yield piece
            if usage_holder is not None:
                usage_holder.append(
                    LLMResult(
                        content="",
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    )
                )
            return
        except Exception as exc:
            last_error = exc
            if started or not _is_retryable(exc) or attempt >= config.LLM_MAX_RETRIES:
                raise LLMUnavailableError() from exc
            delay = config.LLM_RETRY_BACKOFF * (2**attempt)
            delay += random.uniform(0, delay * 0.5)
            logger.warning(
                "LLM stream failed on attempt %d/%d (%s); retrying in %.1fs",
                attempt + 1,
                config.LLM_MAX_RETRIES + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    raise LLMUnavailableError() from last_error