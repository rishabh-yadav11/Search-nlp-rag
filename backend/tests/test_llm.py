"""LLM client tests: generate_answer/stream_answer retry, backoff, exhaustion,
and error classification. The OpenAI-compatible client is mocked; backoff delays
are asserted by recording asyncio.sleep calls instead of actually sleeping."""

import asyncio

import httpx
import openai
import pytest

from app.config import config
from app.llm import LLMResult, LLMUnavailableError, _is_retryable, generate_answer, stream_answer


def _req():
    return httpx.Request("POST", "http://llm.test/v1/chat/completions")


def _api_timeout():
    return openai.APITimeoutError(request=_req())


def _api_conn():
    return openai.APIConnectionError(request=_req())


def _rate_limit():
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=_req()), body=None)


def _server_error():
    return openai.InternalServerError("boom", response=httpx.Response(500, request=_req()), body=None)


def _bad_request():
    return openai.BadRequestError("bad", response=httpx.Response(400, request=_req()), body=None)


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt=0, completion=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Resp:
    def __init__(self, content="hi", prompt=10, completion=5):
        self.usage = _Usage(prompt, completion)
        self.choices = [_Choice(content)]


class _Chunk:
    def __init__(self, content=None, prompt=None, completion=None):
        self.usage = _Usage(prompt, completion) if prompt is not None else None
        self.choices = [] if content is None else [_ChunkChoice(content)]


class _ChunkChoice:
    def __init__(self, content):
        self.delta = _ChunkDelta(content)


class _ChunkDelta:
    def __init__(self, content):
        self.content = content


class _FakeStream:
    """Async iterable of chunks that can raise mid-iteration."""

    def __init__(self, chunks=(), fail_at=None, error=None):
        self._chunks = list(chunks)
        self._fail_at = fail_at
        self._error = error
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._fail_at is not None and self._i == self._fail_at and self._error is not None:
            raise self._error
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


class _FakeCompletions:
    """Mirrors chat.completions.create. ``side_effect`` items may be exceptions,
    responses, or zero-arg callables returning a fresh stream. The last item
    repeats once the list is exhausted, so a single exception models permanent
    failure."""

    def __init__(self, side_effect):
        self.side_effect = list(side_effect)
        self.calls = 0
        self.kwargs = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        self.calls += 1
        item = self.side_effect[min(self.calls - 1, len(self.side_effect) - 1)]
        if callable(item):
            item = item()
        if isinstance(item, Exception):
            raise item
        return item


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    return [piece async for piece in agen]


def _record_sleep(monkeypatch):
    """Replace asyncio.sleep with a recorder; returns the recorded delays."""
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return slept


# --- generate_answer ---


def test_generate_answer_happy_path_propagates_usage():
    client = _FakeClient(_FakeCompletions([_Resp(content="hello", prompt=10, completion=5)]))
    result = _run(generate_answer(client, "PROMPT", "model-x"))

    assert result.content == "hello"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15

    create_kw = client.chat.completions.kwargs[0]
    assert create_kw["model"] == "model-x"
    assert create_kw["max_tokens"] == 1200
    assert create_kw["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert create_kw["timeout"] == config.LLM_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "exc",
    [
        _api_timeout(),
        _api_conn(),
        _rate_limit(),
        _server_error(),
    ],
)
def test_generate_answer_retries_on_retryable_error(monkeypatch, exc):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.5)
    slept = _record_sleep(monkeypatch)

    client = _FakeClient(_FakeCompletions([exc, _Resp(content="retried", prompt=3, completion=2)]))
    result = _run(generate_answer(client, "P", "m"))

    assert result.content == "retried"
    assert result.prompt_tokens == 3
    assert client.chat.completions.calls == 2
    assert slept == [0.5]


def test_generate_answer_exhaustion_raises_and_backs_off(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 3)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.5)
    slept = _record_sleep(monkeypatch)

    client = _FakeClient(_FakeCompletions([_api_timeout()]))
    with pytest.raises(LLMUnavailableError):
        _run(generate_answer(client, "P", "m"))

    assert client.chat.completions.calls == 4
    assert slept == [0.5, 1.0, 2.0]


@pytest.mark.parametrize("exc", [_bad_request(), ValueError("nope")])
def test_generate_answer_non_retryable_raises_immediately(monkeypatch, exc):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    client = _FakeClient(_FakeCompletions([exc]))

    with pytest.raises(LLMUnavailableError):
        _run(generate_answer(client, "P", "m"))

    assert client.chat.completions.calls == 1


# --- _is_retryable ---


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_api_timeout(), True),
        (_api_conn(), True),
        (_rate_limit(), True),
        (_server_error(), True),
        (_bad_request(), False),
        (ValueError("nope"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert _is_retryable(exc) is expected


# --- stream_answer ---


def test_stream_answer_happy_path_with_usage_holder(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)

    def stream():
        return _FakeStream(
            chunks=[
                _Chunk("Hel", 5, 2),
                _Chunk("lo", 3, 1),
                _Chunk(None, 8, 3),
            ]
        )

    client = _FakeClient(_FakeCompletions([stream]))
    holder = []
    pieces = _run(_collect(stream_answer(client, "P", "m", holder)))

    assert pieces == ["Hel", "lo"]
    assert client.chat.completions.calls == 1
    assert holder[0].prompt_tokens == 8
    assert holder[0].completion_tokens == 3


def test_stream_answer_retries_before_first_chunk(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.5)
    slept = _record_sleep(monkeypatch)

    def stream():
        return _FakeStream(chunks=[_Chunk("Hel", 5, 2), _Chunk("lo", 3, 1)])

    client = _FakeClient(_FakeCompletions([_api_conn(), stream]))
    holder = []
    pieces = _run(_collect(stream_answer(client, "P", "m", holder)))

    assert pieces == ["Hel", "lo"]
    assert client.chat.completions.calls == 2
    assert slept == [0.5]


def test_stream_answer_mid_stream_failure_never_retries(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    slept = _record_sleep(monkeypatch)

    client = _FakeClient(
        _FakeCompletions([lambda: _FakeStream(chunks=[_Chunk("part", 5, 2)], fail_at=1, error=_rate_limit())])
    )
    with pytest.raises(LLMUnavailableError):
        _run(_collect(stream_answer(client, "P", "m")))

    assert client.chat.completions.calls == 1
    assert slept == []


def test_stream_answer_exhausts_retries_before_first_chunk(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.5)
    slept = _record_sleep(monkeypatch)

    client = _FakeClient(_FakeCompletions([_api_timeout()]))
    with pytest.raises(LLMUnavailableError):
        _run(_collect(stream_answer(client, "P", "m")))

    assert client.chat.completions.calls == 3
    assert slept == [0.5, 1.0]


def test_stream_answer_without_usage_holder(monkeypatch):
    client = _FakeClient(_FakeCompletions([lambda: _FakeStream(chunks=[_Chunk("hi", 5, 2)])]))
    pieces = _run(_collect(stream_answer(client, "P", "m")))
    assert pieces == ["hi"]
    assert client.chat.completions.calls == 1


def test_stream_answer_zero_usage_when_usage_absent():
    def stream():
        return _FakeStream(chunks=[_Chunk("x")])

    client = _FakeClient(_FakeCompletions([stream]))
    holder = []
    _run(_collect(stream_answer(client, "P", "m", holder)))

    assert isinstance(holder[0], LLMResult)
    assert holder[0].prompt_tokens == 0
    assert holder[0].completion_tokens == 0


def test_generate_answer_raises_without_calling_client_when_no_attempts(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", -1)
    client = _FakeClient(_FakeCompletions([]))
    with pytest.raises(LLMUnavailableError):
        _run(generate_answer(client, "P", "m"))
    assert client.chat.completions.calls == 0


def test_stream_answer_raises_without_calling_client_when_no_attempts(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", -1)
    client = _FakeClient(_FakeCompletions([]))
    with pytest.raises(LLMUnavailableError):
        _run(_collect(stream_answer(client, "P", "m")))
    assert client.chat.completions.calls == 0