from __future__ import annotations

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
import respx
from httpx import AsyncClient

from app.adapters.openai import OpenAIAdapter
from app.core.adapters import get_openai_adapter
from app.core.circuit_breaker import CircuitBreaker, get_circuit_breaker
from app.core.config import Settings, get_settings
from app.core.rate_limiter import RateLimiter, get_rate_limiter
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

    Settings default to retry_max_attempts=1 (i.e. retry effectively
    disabled) so these routing/fallback-focused tests keep their original
    "one HTTP call = one attempt" semantics and don't need real sleep
    delays. Retry behavior itself is covered by tests/test_resilience.py;
    dedicated retry/circuit-breaker tests below override settings per test.
    A fresh, high-threshold CircuitBreaker is used per test so failures in
    one test can never trip a circuit that affects another.
    """
    settings = Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        retry_max_attempts=1,
        retry_base_delay_seconds=0.0,
        retry_max_delay_seconds=0.0,
        circuit_breaker_failure_threshold=1000,
        circuit_breaker_cooldown_seconds=30,
    )
    adapter = OpenAIAdapter(settings, client=httpx.AsyncClient(base_url=settings.openai_base_url))
    circuit_breaker = CircuitBreaker(
        fakeredis.aioredis.FakeRedis(decode_responses=True),
        failure_threshold=settings.circuit_breaker_failure_threshold,
        cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
    )
    # High enough that no routing/fallback-focused test below could ever
    # hit it by accident. Dedicated rate-limit tests override this per test.
    rate_limiter = RateLimiter(
        fakeredis.aioredis.FakeRedis(decode_responses=True),
        requests_per_minute=100_000,
        tokens_per_minute=100_000_000,
    )

    app.dependency_overrides[get_openai_adapter] = lambda: adapter
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_circuit_breaker] = lambda: circuit_breaker
    app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
    # Stashed on the client so individual tests can pre-trip the circuit
    # breaker or swap retry settings without re-plumbing the whole fixture.
    client._test_settings = settings  # type: ignore[attr-defined]
    client._test_circuit_breaker = circuit_breaker  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.pop(get_openai_adapter, None)
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_circuit_breaker, None)
    app.dependency_overrides.pop(get_rate_limiter, None)


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


@pytest.mark.asyncio
@respx.mock
async def test_router_retries_same_model_before_considering_it_failed(chat_client: AsyncClient):
    # Bump retry_max_attempts for just this test (fixture default is 1,
    # i.e. no retry) — proves retry actually happens through the full
    # router path, not just in the lower-level test_resilience.py tests.
    chat_client._test_settings.retry_max_attempts = 3
    chat_client._test_settings.retry_base_delay_seconds = 0.0
    chat_client._test_settings.retry_max_delay_seconds = 0.0

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            _openai_success("gpt-4o"),
        ]
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "fallback_models": ["gpt-4o-mini"], "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 200
    # Response came from gpt-4o itself (the retry succeeded) — fallback to
    # gpt-4o-mini was never needed.
    assert resp.json()["choices"][0]["message"]["content"] == "hi from gpt-4o"
    assert route.call_count == 2  # 1 failed + 1 successful call, same model


@pytest.mark.asyncio
@respx.mock
async def test_router_skips_model_with_open_circuit_and_goes_straight_to_fallback(chat_client: AsyncClient):
    # Override with a fresh, low-threshold circuit breaker just for this
    # test, then trip gpt-4o's circuit via the public API before the
    # request — cleaner than reaching into CircuitBreaker internals.
    low_threshold_cb = CircuitBreaker(fakeredis.aioredis.FakeRedis(decode_responses=True), failure_threshold=1, cooldown_seconds=30)
    await low_threshold_cb.record_failure("gpt-4o")
    app.dependency_overrides[get_circuit_breaker] = lambda: low_threshold_cb

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_openai_success("gpt-4o-mini")
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "fallback_models": ["gpt-4o-mini"], "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi from gpt-4o-mini"
    # Exactly ONE HTTP call was made — gpt-4o was skipped entirely because
    # its circuit was open, not called-and-failed.
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_router_returns_429_when_rate_limited_before_calling_any_model(chat_client: AsyncClient):
    # Override with a 1-request-per-minute limiter just for this test.
    tiny_limiter = RateLimiter(
        fakeredis.aioredis.FakeRedis(decode_responses=True), requests_per_minute=1, tokens_per_minute=100_000
    )
    app.dependency_overrides[get_rate_limiter] = lambda: tiny_limiter

    raw_key = await _create_key(chat_client)

    with respx.mock:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=_openai_success("gpt-4o-mini")
        )

        # 1st request: within the limit, consumes the only unit in the bucket.
        resp1 = await chat_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp1.status_code == 200

        # 2nd request: bucket is empty, should be rejected with 429 —
        # and crucially, NO additional call to OpenAI should have happened.
        resp2 = await chat_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "fallback_models": ["gpt-4o"],  # even with a fallback set...
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp2.status_code == 429
        assert "Retry-After" in resp2.headers
        assert route.call_count == 1  # still just the one call from resp1 — fallback never attempted


@pytest.mark.asyncio
async def test_rate_limit_is_scoped_per_key_not_global(chat_client: AsyncClient):
    tiny_limiter = RateLimiter(
        fakeredis.aioredis.FakeRedis(decode_responses=True), requests_per_minute=1, tokens_per_minute=100_000
    )
    app.dependency_overrides[get_rate_limiter] = lambda: tiny_limiter

    key_a = await _create_key(chat_client)
    key_b = await _create_key(chat_client)

    with respx.mock:
        respx.post("https://api.openai.com/v1/chat/completions").mock(return_value=_openai_success("gpt-4o-mini"))

        resp_a1 = await chat_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert resp_a1.status_code == 200

        resp_a2 = await chat_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert resp_a2.status_code == 429  # key_a is over its own limit

        resp_b1 = await chat_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {key_b}"},
        )
        assert resp_b1.status_code == 200  # key_b has its own untouched bucket
