from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
import respx
from httpx import AsyncClient

from app.adapters.openai import OpenAIAdapter
from app.core.adapters import get_openai_adapter
from app.core.config import Settings
from app.main import app


def _openai_success(model: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{model}",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": f"hi from {model}"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest_asyncio.fixture
async def chat_client(client: AsyncClient) -> AsyncClient:
    """Layers an OpenAIAdapter override on top of the shared `client`
    fixture (which already has the DB dependency overridden with an
    in-memory SQLite session). The adapter's real logic runs; only the
    outbound HTTP call to OpenAI is mocked, via respx, inside each test.
    """
    settings = Settings(openai_api_key="test-key", openai_base_url="https://api.openai.com/v1")
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    app.dependency_overrides[get_openai_adapter] = lambda: adapter
    yield client
    app.dependency_overrides.pop(get_openai_adapter, None)


async def _create_key(client: AsyncClient) -> str:
    resp = await client.post("/v1/keys", json={"name": "chat test key"})
    return resp.json()["api_key"]


@pytest.mark.asyncio
async def test_chat_completions_requires_auth(chat_client: AsyncClient):
    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_success_on_primary_model(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_openai_success("gpt-4o-mini")
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "openai"
    assert body["choices"][0]["message"]["content"] == "hi from gpt-4o-mini"


@pytest.mark.asyncio
@respx.mock
async def test_falls_back_when_primary_fails(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            _openai_success("gpt-4o-mini"),
        ]
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "fallback_models": ["gpt-4o-mini"],
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi from gpt-4o-mini"


@pytest.mark.asyncio
@respx.mock
async def test_falls_back_even_on_non_retryable_failure(chat_client: AsyncClient):
    # Deliberate product decision: fallback triggers on ANY failure,
    # including a 400 that OpenAIAdapter marks retryable=False. A bad
    # request against one model might still be fine against another.
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "unsupported parameter for this model"}}),
            _openai_success("gpt-4o-mini"),
        ]
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "fallback_models": ["gpt-4o-mini"],
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi from gpt-4o-mini"


@pytest.mark.asyncio
@respx.mock
async def test_all_models_failing_returns_502_with_every_attempt(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(500, json={"error": {"message": "internal error"}}),
        ]
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "fallback_models": ["gpt-4o-mini"],
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail["error"] == "All models failed"
    assert len(detail["attempts"]) == 2
    assert detail["attempts"][0] == {
        "model": "gpt-4o",
        "status_code": 429,
        "retryable": True,
        "message": detail["attempts"][0]["message"],
    }
    assert detail["attempts"][1]["model"] == "gpt-4o-mini"
    assert detail["attempts"][1]["status_code"] == 500


@pytest.mark.asyncio
@respx.mock
async def test_no_fallback_models_means_single_attempt(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 502
    assert len(resp.json()["detail"]["attempts"]) == 1
