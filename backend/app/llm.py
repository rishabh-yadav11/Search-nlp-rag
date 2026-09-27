import asyncio
import logging
import random
from dataclasses import dataclass

import openai

from app.config import config

logger = logging.getLogger("llm")

# Upper bound (seconds) on a provider-supplied Retry-After wait so a huge or
# malformed hint can never make a chat/SSE request hang instead of failing fast.
MAX_BACKOFF_SECONDS = 60


def build_messages(prompt: str, system_prompt: str | None = None) -> list[dict]:
    """Chat messages for one turn, with the system prompt in its own role.

    The system prompt must travel as a real ``system`` message: the OpenAI chat
    completions API (openai==1.68.2 ``ChatCompletionSystemMessageParam``) treats
    it as the instruction channel, whereas sending it as a ``user`` message
    leaves it indistinguishable from the untrusted article and conversation text
    sharing that role, so attacker-supplied text can read as an instruction
    (#248). ``system_prompt`` is optional, so a caller with no system text still
    sends a single user message.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be reached, after retries are exhausted.

    ``attempts`` is how many requests were actually sent to the provider. The
    provider bills the prompt of every attempt, retry or not, so a caller that
    charges for a failed call needs the count and not just the fact of failure:
    an outage that burned every retry cost several times a single failed
    request, while zero attempts means the request was never sent and cost
    nothing at all (#280).
    """

    def __init__(self, message: str = "LLM temporarily unavailable", attempts: int = 1):
        super().__init__(message)
        self.attempts = attempts


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


def _retry_after_seconds(exc: Exception) -> float | None:
    """Provider's suggested backoff (seconds) from a 429 ``Retry-After`` header.

    Returns None when the error carries no usable hint, so the caller falls back
    to its own exponential backoff. Only consulted for transient failures, so a
    hinted wait never applies to non-retryable errors."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None) or {}
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.debug("Unparseable Retry-After header %r; ignoring hint", value)
        return None
    if parsed <= 0:
        return None
    return min(parsed, MAX_BACKOFF_SECONDS)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        # Retry only transient failures: 5xx server errors and rate-limit (429).
        # Do NOT retry 4xx client errors (e.g. 400 invalid request) — they won't
        # succeed on retry and would just burn attempts/budget.
        return exc.status_code is not None and (exc.status_code >= 500 or exc.status_code == 429)
    return False


async def generate_answer(llm_client, prompt: str, model: str, system_prompt: str | None = None) -> LLMResult:
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
                messages=build_messages(prompt, system_prompt),
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            usage = response.usage
            if not response.choices:
                raise LLMUnavailableError("LLM returned an empty choices list", attempts=attempt + 1) from None
            return LLMResult(
                content=response.choices[0].message.content or "",
                prompt_tokens=(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                completion_tokens=(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            )
        except LLMUnavailableError:
            raise
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt >= config.LLM_MAX_RETRIES:
                raise LLMUnavailableError(attempts=attempt + 1) from exc
            delay = config.LLM_RETRY_BACKOFF * (2**attempt)
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            delay += random.uniform(0, delay * 0.5)
            logger.warning(
                "LLM call failed on attempt %d/%d (%s); retrying in %.1fs",
                attempt + 1,
                config.LLM_MAX_RETRIES + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    # The loop ran zero times (LLM_MAX_RETRIES < 0), so no request was ever
    # sent and nothing was billed (#280).
    raise LLMUnavailableError(attempts=0) from last_error


async def stream_answer(
    llm_client,
    prompt: str,
    model: str,
    usage_holder: list | None = None,
    system_prompt: str | None = None,
):
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
                messages=build_messages(prompt, system_prompt),
                temperature=config.LLM_TEMPERATURE,
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
                        prompt_tokens=(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                        completion_tokens=(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                    )
                )
            return
        except Exception as exc:
            last_error = exc
            if started or not _is_retryable(exc) or attempt >= config.LLM_MAX_RETRIES:
                raise LLMUnavailableError(attempts=attempt + 1) from exc
            delay = config.LLM_RETRY_BACKOFF * (2**attempt)
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            delay += random.uniform(0, delay * 0.5)
            logger.warning(
                "LLM stream failed on attempt %d/%d (%s); retrying in %.1fs",
                attempt + 1,
                config.LLM_MAX_RETRIES + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    # The loop ran zero times (LLM_MAX_RETRIES < 0): no request was sent, so
    # nothing was billed (#280).
    raise LLMUnavailableError(attempts=0) from last_error
