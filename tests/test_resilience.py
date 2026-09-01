import fakeredis.aioredis
import httpx
import pytest
import respx

from app.adapters.base import ProviderError
from app.adapters.openai import OpenAIAdapter
from app.core.circuit_breaker import CircuitBreaker
from app.core.config import Settings
from app.core.resilience import CircuitOpenError, call_model, call_model_stream
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, ChunkDelta, Message


def fast_settings(**overrides) -> Settings:
    defaults = dict(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        retry_max_attempts=3,
        retry_base_delay_seconds=0.0,  # no real sleeping in tests
        retry_max_delay_seconds=0.0,
        circuit_breaker_failure_threshold=100,  # effectively disabled unless overridden
        circuit_breaker_cooldown_seconds=30,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_request(model: str = "gpt-4o") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


def _success_response(model: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.mark.asyncio
@respx.mock
async def test_succeeds_on_first_try_no_retry_needed():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_success_response("gpt-4o")
    )
    settings = fast_settings()
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    result = await call_model(adapter, cb, make_request(), settings)

    assert result.choices[0].message.content == "ok"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_retries_retryable_failure_then_succeeds():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(500, json={"error": {"message": "server error"}}),
            _success_response("gpt-4o"),
        ]
    )
    settings = fast_settings(retry_max_attempts=3)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    result = await call_model(adapter, cb, make_request(), settings)

    assert result.choices[0].message.content == "ok"
    assert route.call_count == 3  # 2 failed attempts + 1 success, all against the SAME model


@pytest.mark.asyncio
@respx.mock
async def test_non_retryable_failure_raises_immediately_no_retry():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    settings = fast_settings(retry_max_attempts=3)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    with pytest.raises(ProviderError) as exc_info:
        await call_model(adapter, cb, make_request(), settings)

    assert exc_info.value.status_code == 400
    assert route.call_count == 1  # no retry attempted for a non-retryable error


@pytest.mark.asyncio
@respx.mock
async def test_exhausts_retries_and_raises_last_error():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "always rate limited"}})
    )
    settings = fast_settings(retry_max_attempts=3)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    with pytest.raises(ProviderError) as exc_info:
        await call_model(adapter, cb, make_request(), settings)

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
@respx.mock
async def test_failure_recorded_on_circuit_breaker():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    settings = fast_settings(retry_max_attempts=1)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=2, cooldown_seconds=30)

    with pytest.raises(ProviderError):
        await call_model(adapter, cb, make_request(), settings)
    assert await cb.is_open("gpt-4o") is False  # 1 failure, threshold is 2

    with pytest.raises(ProviderError):
        await call_model(adapter, cb, make_request(), settings)
    assert await cb.is_open("gpt-4o") is True  # 2nd failure trips it


@pytest.mark.asyncio
@respx.mock
async def test_success_recorded_on_circuit_breaker_resets_it():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            _success_response("gpt-4o"),
        ]
    )
    settings = fast_settings(retry_max_attempts=2)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=5, cooldown_seconds=30)

    await call_model(adapter, cb, make_request(), settings)

    assert (await cb.snapshot("gpt-4o")).failure_count == 0


@pytest.mark.asyncio
async def test_open_circuit_skips_call_entirely():
    settings = fast_settings()
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=1, cooldown_seconds=30)
    await cb.record_failure("gpt-4o")  # trip the circuit before any call

    with pytest.raises(CircuitOpenError):
        # No respx mock registered at all — if this made a real HTTP call
        # it would raise a connection error, not CircuitOpenError. Passing
        # proves the call was skipped entirely.
        await call_model(adapter, cb, make_request(), settings)


def _sse(*lines: str) -> str:
    return "".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n"


@pytest.mark.asyncio
@respx.mock
async def test_stream_succeeds_on_first_try_no_retry_needed():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse('{"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}'),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    settings = fast_settings()
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    chunks = [c async for c in call_model_stream(adapter, cb, make_request(), settings)]

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hi"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_stream_retries_retryable_failure_before_first_chunk_then_succeeds():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(
                200,
                content=_sse('{"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}'),
                headers={"Content-Type": "text/event-stream"},
            ),
        ]
    )
    settings = fast_settings(retry_max_attempts=3)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    chunks = [c async for c in call_model_stream(adapter, cb, make_request(), settings)]

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "ok"
    assert route.call_count == 2  # 1 failed pre-chunk attempt + 1 success, same model


@pytest.mark.asyncio
@respx.mock
async def test_stream_non_retryable_failure_before_first_chunk_raises_immediately():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    settings = fast_settings(retry_max_attempts=3)
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)

    with pytest.raises(ProviderError) as exc_info:
        async for _ in call_model_stream(adapter, cb, make_request(), settings):
            pass

    assert exc_info.value.status_code == 400
    assert route.call_count == 1  # no retry for a non-retryable error


@pytest.mark.asyncio
@respx.mock
async def test_stream_open_circuit_skips_call_entirely():
    settings = fast_settings()
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=1, cooldown_seconds=30)
    await cb.record_failure("gpt-4o")

    with pytest.raises(CircuitOpenError):
        # No respx route registered at all — if this made a real call it
        # would raise a connection error, not CircuitOpenError.
        async for _ in call_model_stream(adapter, cb, make_request(), settings):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_stream_no_retry_once_past_first_chunk_even_on_transport_error():
    # Simulates a connection that succeeds, yields one real chunk, then
    # drops — the generator raises mid-iteration. call_model_stream must
    # NOT retry this: the caller already received the first chunk and
    # can't un-send it to a client downstream.
    class _DropsAfterFirstChunkAdapter:
        name = "openai"
        call_count = 0

        async def chat_completion_stream(self, request):
            _DropsAfterFirstChunkAdapter.call_count += 1
            yield ChatCompletionChunk(
                id="c1", model="gpt-4o", provider="openai",
                choices=[ChunkChoice(index=0, delta=ChunkDelta(content="partial"), finish_reason=None)],
            )
            raise ProviderError("connection dropped mid-stream", provider="openai", status_code=503, retryable=True)

    settings = fast_settings()
    cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=100, cooldown_seconds=30)
    adapter = _DropsAfterFirstChunkAdapter()

    received = []
    with pytest.raises(ProviderError):
        async for chunk in call_model_stream(adapter, cb, make_request(), settings):
            received.append(chunk)

    assert len(received) == 1  # got the partial chunk
    assert adapter.call_count == 1  # never retried — only one call to the adapter ever happened
