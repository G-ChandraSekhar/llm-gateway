from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import redis.asyncio as redis

from app.core.config import get_settings

# Reads current state and, if it's OPEN and the cooldown has elapsed,
# atomically transitions it to HALF_OPEN and lets exactly one trial call
# through. This matters under concurrency: if two requests arrive the
# instant the cooldown expires, only ONE of them may get the single
# trial call — the other must still be blocked until that trial resolves
# (via record_success or record_failure), not just until the OPEN state
# happens to still be visible. So HALF_OPEN blocks everyone except the
# single call that performed the open->half_open transition itself;
# every other read of an already-half_open state returns "blocked".
#
# KEYS[1] = circuit key
# ARGV[1] = now (unix seconds, float)
# ARGV[2] = cooldown_seconds
# Returns: 1 if the call should be skipped (circuit open or a trial is
# already in flight), 0 if this call may proceed.
_IS_OPEN_SCRIPT = """
local now = tonumber(ARGV[1])
local cooldown = tonumber(ARGV[2])

local state = redis.call('HGET', KEYS[1], 'state')
if state == false or state == 'closed' then
    return 0
end

if state == 'half_open' then
    -- A trial call is already in flight for this circuit; don't let a
    -- second one through concurrently.
    return 1
end

-- state == 'open'
local opened_at = tonumber(redis.call('HGET', KEYS[1], 'opened_at'))
if opened_at == nil then
    return 0
end

if now - opened_at >= cooldown then
    redis.call('HSET', KEYS[1], 'state', 'half_open')
    return 0
end
return 1
"""

# Increments the failure count and opens the circuit if it crosses the
# threshold, OR — if the circuit was in HALF_OPEN (a trial call that just
# failed) — reopens it immediately and restarts the cooldown, without
# needing failure_count to reach the threshold again.
#
# KEYS[1] = circuit key
# ARGV[1] = now (unix seconds, float)
# ARGV[2] = failure_threshold
# ARGV[3] = key TTL in seconds
_RECORD_FAILURE_SCRIPT = """
local now = tonumber(ARGV[1])
local threshold = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local state = redis.call('HGET', KEYS[1], 'state')

if state == 'half_open' then
    redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', now)
    redis.call('EXPIRE', KEYS[1], ttl)
    return
end

local count = tonumber(redis.call('HGET', KEYS[1], 'failure_count'))
if count == nil then count = 0 end
count = count + 1

if count >= threshold then
    redis.call('HSET', KEYS[1], 'state', 'open', 'failure_count', count, 'opened_at', now)
else
    redis.call('HSET', KEYS[1], 'state', 'closed', 'failure_count', count)
end
redis.call('EXPIRE', KEYS[1], ttl)
"""


@dataclass
class CircuitSnapshot:
    state: str
    failure_count: int


class CircuitBreaker:
    """Tracks failures per key (here, per model name) and opens that key's
    circuit after `failure_threshold` consecutive failures, so a model
    that's clearly down stops being called — including retries — until a
    cooldown elapses.

    Redis-backed (was in-memory through Day 5 — see tasks/todo.md for that
    history) via the same atomic-Lua-script pattern as the Day 7 rate
    limiter, so state is now shared across gateway instances/processes,
    not per-process. Public interface (is_open / record_success /
    record_failure) is unchanged from the in-memory version — everything
    calling this class didn't need to change, only what's inside it.

    Known edge case, not fixed: if the single trial call that gets let
    through during HALF_OPEN never resolves (the process crashes mid-call
    before calling record_success/record_failure), the circuit is stuck
    in HALF_OPEN indefinitely — nothing times it out. A production version
    would want a lease/TTL on the half-open trial itself; out of scope
    here, and unlikely to matter for a single-gateway portfolio deployment.
    """

    def __init__(self, redis_client: redis.Redis, failure_threshold: int, cooldown_seconds: float, key_ttl_seconds: int | None = None):
        self._redis = redis_client
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._key_ttl = key_ttl_seconds or max(int(cooldown_seconds) * 10, 3600)
        self._is_open_script = self._redis.register_script(_IS_OPEN_SCRIPT)
        self._record_failure_script = self._redis.register_script(_RECORD_FAILURE_SCRIPT)

    def _key(self, model: str) -> str:
        return f"circuit:{model}"

    async def is_open(self, key: str) -> bool:
        """True if a call for this key should be skipped right now."""
        result = await self._is_open_script(keys=[self._key(key)], args=[time.time(), self._cooldown_seconds])
        return bool(int(result))

    async def record_success(self, key: str) -> None:
        # Unconditional reset — a single HSET is already atomic as one
        # Redis command, no Lua needed for this one.
        await self._redis.hset(self._key(key), mapping={"state": "closed", "failure_count": 0, "opened_at": ""})
        await self._redis.expire(self._key(key), self._key_ttl)

    async def record_failure(self, key: str) -> None:
        await self._record_failure_script(
            keys=[self._key(key)], args=[time.time(), self._failure_threshold, self._key_ttl]
        )

    async def snapshot(self, key: str) -> CircuitSnapshot:
        """Read-only view of a circuit's current state — for tests and
        future observability endpoints, not used in the request path.
        """
        data = await self._redis.hgetall(self._key(key))
        state = data.get("state", "closed")
        failure_count = int(data.get("failure_count", 0))
        return CircuitSnapshot(state=state, failure_count=failure_count)


@lru_cache
def get_redis_client_for_circuit_breaker() -> redis.Redis:
    # Separate client instance from the rate limiter's, even though both
    # point at the same REDIS_URL — keeps the two subsystems decoupled
    # (e.g. free to move to different Redis instances/DBs later) without
    # sharing connection-pool lifecycle.
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_circuit_breaker() -> CircuitBreaker:
    settings = get_settings()
    return CircuitBreaker(
        get_redis_client_for_circuit_breaker(),
        settings.circuit_breaker_failure_threshold,
        settings.circuit_breaker_cooldown_seconds,
    )
