from __future__ import annotations

import fakeredis.aioredis
import pytest

from app.core.rate_limiter import RateLimiter


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.mark.asyncio
async def test_allows_requests_within_limit(redis_client):
    limiter = RateLimiter(redis_client, requests_per_minute=5, tokens_per_minute=10_000)

    for _ in range(5):
        result = await limiter.check("key-a", estimated_tokens=10)
        assert result.allowed is True


@pytest.mark.asyncio
async def test_blocks_once_requests_bucket_is_empty(redis_client):
    limiter = RateLimiter(redis_client, requests_per_minute=3, tokens_per_minute=10_000)

    for _ in range(3):
        assert (await limiter.check("key-a", estimated_tokens=10)).allowed is True

    result = await limiter.check("key-a", estimated_tokens=10)
    assert result.allowed is False
    assert result.reason == 1  # requests bucket, not tokens
    assert result.retry_after_seconds > 0


@pytest.mark.asyncio
async def test_blocks_once_tokens_bucket_is_empty(redis_client):
    limiter = RateLimiter(redis_client, requests_per_minute=1000, tokens_per_minute=100)

    result = await limiter.check("key-a", estimated_tokens=80)
    assert result.allowed is True

    # Only 20 tokens left in the bucket; this request wants 30.
    result = await limiter.check("key-a", estimated_tokens=30)
    assert result.allowed is False
    assert result.reason == 2  # tokens bucket


@pytest.mark.asyncio
async def test_rejected_request_does_not_consume_the_other_bucket(redis_client):
    # A request rejected for being over the TOKEN budget must not still
    # burn a unit of the REQUEST budget — otherwise a single oversized
    # request could exhaust the request bucket for free.
    limiter = RateLimiter(redis_client, requests_per_minute=5, tokens_per_minute=100)

    rejected = await limiter.check("key-a", estimated_tokens=1000)
    assert rejected.allowed is False
    assert rejected.reason == 2

    # Requests bucket should still be untouched — full 5 available.
    for _ in range(5):
        assert (await limiter.check("key-a", estimated_tokens=1)).allowed is True


@pytest.mark.asyncio
async def test_different_keys_have_independent_buckets(redis_client):
    limiter = RateLimiter(redis_client, requests_per_minute=1, tokens_per_minute=10_000)

    assert (await limiter.check("key-a", estimated_tokens=1)).allowed is True
    assert (await limiter.check("key-a", estimated_tokens=1)).allowed is False
    # key-b has never been used, has its own full bucket
    assert (await limiter.check("key-b", estimated_tokens=1)).allowed is True


@pytest.mark.asyncio
async def test_bucket_refills_over_time(redis_client, monkeypatch):
    limiter = RateLimiter(redis_client, requests_per_minute=60, tokens_per_minute=10_000)  # 1/sec refill

    current_time = [1000.0]
    monkeypatch.setattr("app.core.rate_limiter.time.time", lambda: current_time[0])

    for _ in range(60):
        assert (await limiter.check("key-a", estimated_tokens=1)).allowed is True
    assert (await limiter.check("key-a", estimated_tokens=1)).allowed is False  # bucket empty

    current_time[0] += 5  # 5 seconds pass -> ~5 requests worth refilled
    result = await limiter.check("key-a", estimated_tokens=1)
    assert result.allowed is True
