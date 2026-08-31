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
