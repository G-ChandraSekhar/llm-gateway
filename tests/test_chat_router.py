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
from tests.conftest import ADMIN_HEADERS, TEST_ADMIN_KEY


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
        admin_api_key=TEST_ADMIN_KEY,
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
    resp = await client.post("/v1/keys", json={"name": "chat test key"}, headers=ADMIN_HEADERS)
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


@pytest.mark.asyncio
@respx.mock
async def test_over_budget_key_gets_402_before_calling_any_model(chat_client: AsyncClient):
    create_resp = await chat_client.post(
        "/v1/keys", json={"name": "broke", "budget_limit_usd": 0.01}, headers=ADMIN_HEADERS
    )
    raw_key = create_resp.json()["api_key"]

    # Manually push spend over the limit by hitting the DB directly through
    # the same in-memory session the app uses (no endpoint exists yet to
    # set spend directly — that's the point, spend only moves via real
    # usage, so we simulate "already spent" the same way a real prior
    # request would have left it).
    from sqlalchemy import select

    from app.models.api_key import APIKey

    result = await chat_client._test_db_session.execute(select(APIKey).where(APIKey.prefix == create_resp.json()["prefix"]))
    key_row = result.scalar_one()
    key_row.spent_micros = 20_000  # $0.02, over the $0.01 limit
    await chat_client._test_db_session.commit()

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_openai_success("gpt-4o-mini")
    )

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    assert resp.status_code == 402
    assert resp.json()["detail"]["error"] == "Budget exceeded"
    assert route.call_count == 0  # no model was ever called


@pytest.mark.asyncio
@respx.mock
async def test_successful_call_records_real_spend(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-spend",
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
            },
        )
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 200

    me_resp = await chat_client.get("/v1/keys/me", headers={"Authorization": f"Bearer {raw_key}"})
    # 1000 prompt @ $0.15/M + 1000 completion @ $0.60/M = (150 + 600) / 1e6 = $0.00075
    assert me_resp.json()["spent_usd"] == pytest.approx(0.00075, abs=1e-6)


@pytest.mark.asyncio
@respx.mock
async def test_spend_recorded_against_the_model_that_actually_served_not_the_primary(chat_client: AsyncClient):
    # Primary model gpt-4o fails, fallback gpt-4o-mini succeeds — spend
    # must use gpt-4o-mini's (cheaper) pricing, not gpt-4o's.
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500, json={"error": {"message": "down"}}),
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fallback-spend",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
                },
            ),
        ]
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "fallback_models": ["gpt-4o-mini"], "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 200

    me_resp = await chat_client.get("/v1/keys/me", headers={"Authorization": f"Bearer {raw_key}"})
    # Must match gpt-4o-mini's pricing ($0.00075), NOT gpt-4o's (would be $0.0125)
    assert me_resp.json()["spent_usd"] == pytest.approx(0.00075, abs=1e-6)


def _sse_body_str(*lines: str) -> str:
    return "".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n"


@pytest.mark.asyncio
@respx.mock
async def test_stream_success_forwards_chunks_and_ends_with_done(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse_body_str(
                '{"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
                '{"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi!"},"finish_reason":"stop"}]}',
                '{"id":"c1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    raw_key = await _create_key(chat_client)

    async with chat_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": f"Bearer {raw_key}"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = b"".join([chunk async for chunk in resp.aiter_bytes()]).decode()

    assert body.count("data: ") == 4  # role chunk, content chunk, usage chunk, [DONE]
    assert '"content":"Hi!"' in body
    assert body.strip().endswith("data: [DONE]")


@pytest.mark.asyncio
@respx.mock
async def test_stream_falls_back_before_first_chunk(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500, json={"error": {"message": "down"}}),
            httpx.Response(
                200,
                content=_sse_body_str(
                    '{"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"fallback worked"},"finish_reason":"stop"}]}'
                ),
                headers={"Content-Type": "text/event-stream"},
            ),
        ]
    )
    raw_key = await _create_key(chat_client)

    async with chat_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "fallback_models": ["gpt-4o-mini"],
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {raw_key}"},
    ) as resp:
        assert resp.status_code == 200
        body = b"".join([chunk async for chunk in resp.aiter_bytes()]).decode()

    assert "fallback worked" in body


@pytest.mark.asyncio
@respx.mock
async def test_stream_all_models_failing_before_first_chunk_is_a_real_502(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "down"}})
    )
    raw_key = await _create_key(chat_client)

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    # A real HTTP 502, NOT a 200 with an error event inside SSE — because
    # nothing was ever streamed to the client yet.
    assert resp.status_code == 502
    assert resp.json()["detail"]["error"] == "All models failed"


@pytest.mark.asyncio
@respx.mock
async def test_stream_records_spend_from_final_usage_chunk(chat_client: AsyncClient):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse_body_str(
                '{"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}',
                '{"id":"c1","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":1000,"total_tokens":2000}}',
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    raw_key = await _create_key(chat_client)

    async with chat_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": f"Bearer {raw_key}"},
    ) as resp:
        async for _ in resp.aiter_bytes():
            pass

    me_resp = await chat_client.get("/v1/keys/me", headers={"Authorization": f"Bearer {raw_key}"})
    assert me_resp.json()["spent_usd"] == pytest.approx(0.00075, abs=1e-6)


@pytest.mark.asyncio
@respx.mock
async def test_stream_rate_limited_gets_429_before_any_streaming(chat_client: AsyncClient):
    tiny_limiter = RateLimiter(
        fakeredis.aioredis.FakeRedis(decode_responses=True), requests_per_minute=1, tokens_per_minute=100_000
    )
    app.dependency_overrides[get_rate_limiter] = lambda: tiny_limiter

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_openai_success("gpt-4o-mini")
    )
    raw_key = await _create_key(chat_client)

    # Use up the only request in the bucket (non-streaming, simpler).
    await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )

    resp = await chat_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 429
