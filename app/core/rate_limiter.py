from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import redis.asyncio as redis
from fastapi import Depends, HTTPException, status

from app.core.auth import get_current_api_key
from app.core.config import get_settings
from app.core.token_estimate import estimate_request_tokens
from app.models.api_key import APIKey
from app.schemas.chat import ChatCompletionRequest

# Atomic in a single round trip: refills both buckets (requests, tokens)
# based on elapsed time, then only commits the consumption if BOTH have
# enough headroom. If either is short, neither bucket is touched — so a
# request that's rejected for being over the token budget doesn't also
# waste a unit of the request budget.
#
# KEYS[1] = requests bucket hash key
# KEYS[2] = tokens bucket hash key
# ARGV[1] = now (unix seconds, float)
# ARGV[2] = requests bucket capacity
# ARGV[3] = requests refill per second
# ARGV[4] = tokens bucket capacity
# ARGV[5] = tokens refill per second
# ARGV[6] = estimated token cost of this request
# ARGV[7] = bucket key TTL in seconds (so idle keys expire instead of
#           accumulating in Redis forever)
_TOKEN_BUCKET_SCRIPT = """
local now = tonumber(ARGV[1])
local req_cap = tonumber(ARGV[2])
local req_refill = tonumber(ARGV[3])
local tok_cap = tonumber(ARGV[4])
local tok_refill = tonumber(ARGV[5])
local tok_cost = tonumber(ARGV[6])
local ttl = tonumber(ARGV[7])

local function refill(key, capacity, refill_rate)
    local data = redis.call('HMGET', key, 'level', 'ts')
    local level = tonumber(data[1])
    local ts = tonumber(data[2])
    if level == nil then
        level = capacity
        ts = now
    end
    local elapsed = now - ts
    if elapsed < 0 then elapsed = 0 end
    level = math.min(capacity, level + elapsed * refill_rate)
    return level
end

local req_level = refill(KEYS[1], req_cap, req_refill)
local tok_level = refill(KEYS[2], tok_cap, tok_refill)

local req_ok = req_level >= 1
local tok_ok = tok_level >= tok_cost

if req_ok and tok_ok then
    req_level = req_level - 1
    tok_level = tok_level - tok_cost
    redis.call('HMSET', KEYS[1], 'level', req_level, 'ts', now)
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('HMSET', KEYS[2], 'level', tok_level, 'ts', now)
    redis.call('EXPIRE', KEYS[2], ttl)
    return {1, tostring(req_level), tostring(tok_level), 0}
else
    local reason = 1
    if not req_ok then reason = 1 else reason = 2 end
    return {0, tostring(req_level), tostring(tok_level), reason}
end
"""


@dataclass
class RateLimitResult:
    allowed: bool
    requests_remaining: float
    tokens_remaining: float
    # 0 if allowed; 1 if the requests/min bucket was empty; 2 if the
    # tokens/min bucket was empty (checked in that order).
    reason: int = 0
    retry_after_seconds: float = 0.0


class RateLimiter:
    """Redis-backed token bucket, per API key, for both requests/min and
    tokens/min. The token cost is an ESTIMATE made before the call (see
    estimate_request_tokens) — this limiter is about protecting the
    gateway and upstream from being hammered, not about billing accuracy.
    Day 8's budget tracker uses the provider's real post-call token usage
    for that; the two numbers are allowed to disagree.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        requests_per_minute: int,
        tokens_per_minute: int,
        bucket_ttl_seconds: int = 120,
    ):
        self._redis = redis_client
        self._requests_capacity = requests_per_minute
        self._requests_refill_per_sec = requests_per_minute / 60
        self._tokens_capacity = tokens_per_minute
        self._tokens_refill_per_sec = tokens_per_minute / 60
        self._bucket_ttl = bucket_ttl_seconds
        self._script = self._redis.register_script(_TOKEN_BUCKET_SCRIPT)

    async def check(self, api_key_id: str, estimated_tokens: int) -> RateLimitResult:
        now = time.time()
        req_key = f"ratelimit:{api_key_id}:requests"
        tok_key = f"ratelimit:{api_key_id}:tokens"

        raw = await self._script(
            keys=[req_key, tok_key],
            args=[
                now,
                self._requests_capacity,
                self._requests_refill_per_sec,
                self._tokens_capacity,
                self._tokens_refill_per_sec,
                estimated_tokens,
                self._bucket_ttl,
            ],
        )
        allowed, req_level, tok_level, reason = raw
        req_level = float(req_level)
        tok_level = float(tok_level)

        if allowed:
            return RateLimitResult(allowed=True, requests_remaining=req_level, tokens_remaining=tok_level)

        reason = int(reason)
        if reason == 1:
            deficit = 1 - req_level
            retry_after = deficit / self._requests_refill_per_sec if self._requests_refill_per_sec > 0 else 60.0
        else:
            deficit = estimated_tokens - tok_level
            retry_after = deficit / self._tokens_refill_per_sec if self._tokens_refill_per_sec > 0 else 60.0

        return RateLimitResult(
            allowed=False,
            requests_remaining=req_level,
            tokens_remaining=tok_level,
            reason=reason,
            retry_after_seconds=max(retry_after, 0.0),
        )


@lru_cache
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_rate_limiter() -> RateLimiter:
    settings = get_settings()
    return RateLimiter(
        get_redis_client(),
        settings.rate_limit_requests_per_minute,
        settings.rate_limit_tokens_per_minute,
    )

async def enforce_rate_limit(
    body: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """FastAPI dependency: one rate-limit check per incoming request, not
    per model attempt. A key over its limit is over its limit regardless
    of which OpenAI model it's asking for — trying a different model
    (Day 4's fallback) wouldn't help, since it's the same key hitting the
    same bucket. So this raises before any model is ever attempted, and
    Day 4/5's retry/fallback logic never runs for a rate-limited request.
    """
    estimated_tokens = estimate_request_tokens(body)
    result = await limiter.check(api_key.id, estimated_tokens)

    if not result.allowed:
        reason_text = "requests-per-minute limit exceeded" if result.reason == 1 else "tokens-per-minute limit exceeded"
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(round(result.retry_after_seconds, 2))},
            detail={
                "error": reason_text,
                "retry_after_seconds": round(result.retry_after_seconds, 2),
            },
        )
