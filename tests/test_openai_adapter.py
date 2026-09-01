import httpx
import pytest
import respx

from app.adapters.base import ProviderError
from app.adapters.openai import OpenAIAdapter
from app.core.config import Settings
from app.schemas.chat import ChatCompletionRequest, Message


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_base_url="https://api.openai.com/v1")


@pytest.fixture
def adapter(settings: Settings) -> OpenAIAdapter:
    client = httpx.AsyncClient(base_url=settings.openai_base_url)
    return OpenAIAdapter(settings, client=client)


def make_request(**overrides) -> ChatCompletionRequest:
    defaults = dict(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="hello")],
    )
    defaults.update(overrides)
    return ChatCompletionRequest(**defaults)


@pytest.mark.asyncio
@respx.mock
async def test_success_maps_response(adapter: OpenAIAdapter):
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-abc",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )
    )

    result = await adapter.chat_completion(make_request())

    sent_body = route.calls[0].request.content
    assert b'"model":"gpt-4o-mini"' in sent_body

    assert result.provider == "openai"
    assert result.choices[0].message.content == "hi there"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.total_tokens == 8


@pytest.mark.asyncio
@respx.mock
async def test_429_is_retryable(adapter: OpenAIAdapter):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )

    with pytest.raises(ProviderError) as exc_info:
        await adapter.chat_completion(make_request())

    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
@respx.mock
async def test_401_is_not_retryable(adapter: OpenAIAdapter):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid api key"}})
    )

    with pytest.raises(ProviderError) as exc_info:
        await adapter.chat_completion(make_request())

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_timeout_is_retryable(adapter: OpenAIAdapter):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("connect timed out")
    )

    with pytest.raises(ProviderError) as exc_info:
        await adapter.chat_completion(make_request())

    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_malformed_2xx_response_is_not_retryable(adapter: OpenAIAdapter):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )

    with pytest.raises(ProviderError) as exc_info:
        await adapter.chat_completion(make_request())

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
@respx.mock
async def test_fallback_models_field_is_never_sent_to_openai(adapter: OpenAIAdapter):
    # fallback_models is a gateway-only field the router (Day 4) will use to
    # retry with an alternate OpenAI model (e.g. gpt-4o -> gpt-4o-mini). The
    # adapter itself must never forward it upstream.
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    await adapter.chat_completion(make_request(model="gpt-4o", fallback_models=["gpt-4o-mini"]))

    sent_body = route.calls[0].request.content
    assert b"fallback_models" not in sent_body
    assert b'"model":"gpt-4o"' in sent_body


@pytest.mark.asyncio
@respx.mock
async def test_stream_yields_role_content_and_usage_chunks(adapter: OpenAIAdapter):
    sse_body = (
        'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"c1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        'data: [DONE]\n\n'
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body, headers={"Content-Type": "text/event-stream"})
    )

    chunks = [c async for c in adapter.chat_completion_stream(make_request())]

    assert len(chunks) == 4
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.content == "Hi"
    assert chunks[2].choices[0].finish_reason == "stop"
    assert chunks[3].choices == []
    assert chunks[3].usage.total_tokens == 7


@pytest.mark.asyncio
@respx.mock
async def test_stream_request_includes_stream_options_for_usage(adapter: OpenAIAdapter):
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content="data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
    )

    async for _ in adapter.chat_completion_stream(make_request()):
        pass

    sent = route.calls[0].request.content
    assert b'"stream":true' in sent
    assert b'"include_usage":true' in sent


@pytest.mark.asyncio
@respx.mock
async def test_stream_rejects_before_yielding_anything_on_error_status(adapter: OpenAIAdapter):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )

    gen = adapter.chat_completion_stream(make_request())
    with pytest.raises(ProviderError) as exc_info:
        await gen.__anext__()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_stream_with_no_chunks_at_all_just_ends(adapter: OpenAIAdapter):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content="data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
    )

    chunks = [c async for c in adapter.chat_completion_stream(make_request())]
    assert chunks == []


@pytest.mark.asyncio
@respx.mock
async def test_stream_ignores_malformed_sse_line_mid_stream(adapter: OpenAIAdapter):
    sse_body = (
        'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        "data: not valid json at all\n\n"
        'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"still ok"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_body, headers={"Content-Type": "text/event-stream"})
    )

    chunks = [c async for c in adapter.chat_completion_stream(make_request())]
    assert len(chunks) == 2  # the malformed line was skipped, not fatal
    assert chunks[1].choices[0].delta.content == "still ok"
