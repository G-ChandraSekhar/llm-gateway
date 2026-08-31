from __future__ import annotations

import fakeredis.aioredis
import pytest

from app.core.circuit_breaker import CircuitBreaker


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def breaker(redis_client):
    return CircuitBreaker(redis_client, failure_threshold=3, cooldown_seconds=10)


@pytest.mark.asyncio
async def test_starts_closed(breaker: CircuitBreaker):
    assert await breaker.is_open("gpt-4o") is False
    snap = await breaker.snapshot("gpt-4o")
    assert snap.state == "closed"


@pytest.mark.asyncio
async def test_opens_after_threshold_consecutive_failures(breaker: CircuitBreaker):
    await breaker.record_failure("gpt-4o")
    await breaker.record_failure("gpt-4o")
    assert await breaker.is_open("gpt-4o") is False  # 2 failures, threshold is 3

    await breaker.record_failure("gpt-4o")
    assert await breaker.is_open("gpt-4o") is True


@pytest.mark.asyncio
async def test_success_resets_failure_count(breaker: CircuitBreaker):
    await breaker.record_failure("gpt-4o")
    await breaker.record_failure("gpt-4o")
    await breaker.record_success("gpt-4o")

    snap = await breaker.snapshot("gpt-4o")
    assert snap.failure_count == 0

    await breaker.record_failure("gpt-4o")
    await breaker.record_failure("gpt-4o")
    assert await breaker.is_open("gpt-4o") is False  # only 2 since the reset


@pytest.mark.asyncio
async def test_different_models_have_independent_circuits(redis_client):
    breaker = CircuitBreaker(redis_client, failure_threshold=2, cooldown_seconds=10)

    await breaker.record_failure("gpt-4o")
    await breaker.record_failure("gpt-4o")

    assert await breaker.is_open("gpt-4o") is True
    assert await breaker.is_open("gpt-4o-mini") is False


@pytest.mark.asyncio
async def test_stays_open_until_cooldown_elapses(redis_client, monkeypatch):
    current_time = [1000.0]
    monkeypatch.setattr("app.core.circuit_breaker.time.time", lambda: current_time[0])
    breaker = CircuitBreaker(redis_client, failure_threshold=1, cooldown_seconds=30)

    await breaker.record_failure("gpt-4o")
    assert await breaker.is_open("gpt-4o") is True

    current_time[0] += 29
    assert await breaker.is_open("gpt-4o") is True  # still within cooldown

    current_time[0] += 2  # now 31s elapsed, past the 30s cooldown
    assert await breaker.is_open("gpt-4o") is False


@pytest.mark.asyncio
async def test_half_open_success_closes_circuit(redis_client, monkeypatch):
    current_time = [1000.0]
    monkeypatch.setattr("app.core.circuit_breaker.time.time", lambda: current_time[0])
    breaker = CircuitBreaker(redis_client, failure_threshold=1, cooldown_seconds=10)

    await breaker.record_failure("gpt-4o")
    current_time[0] += 11
    assert await breaker.is_open("gpt-4o") is False  # transitions to half-open, trial allowed

    await breaker.record_success("gpt-4o")

    snap = await breaker.snapshot("gpt-4o")
    assert snap.state == "closed"
    assert await breaker.is_open("gpt-4o") is False


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit_and_restarts_cooldown(redis_client, monkeypatch):
    current_time = [1000.0]
    monkeypatch.setattr("app.core.circuit_breaker.time.time", lambda: current_time[0])
    breaker = CircuitBreaker(redis_client, failure_threshold=1, cooldown_seconds=10)

    await breaker.record_failure("gpt-4o")
    current_time[0] += 11
    assert await breaker.is_open("gpt-4o") is False  # half-open trial allowed

    await breaker.record_failure("gpt-4o")  # trial call failed
    assert await breaker.is_open("gpt-4o") is True

    current_time[0] += 9
    assert await breaker.is_open("gpt-4o") is True  # cooldown restarted, not the original clock

    current_time[0] += 2  # 11s since the reopen
    assert await breaker.is_open("gpt-4o") is False


@pytest.mark.asyncio
async def test_concurrent_half_open_transition_only_permits_one_trial(redis_client, monkeypatch):
    # The scenario this script's atomicity exists to prevent: two
    # requests arriving the instant the cooldown elapses. Only the first
    # is_open() call may get the trial; a second one immediately after
    # (still half_open, trial not yet resolved) must still be blocked —
    # otherwise both would hit the still-possibly-down model at once.
    current_time = [1000.0]
    monkeypatch.setattr("app.core.circuit_breaker.time.time", lambda: current_time[0])
    breaker = CircuitBreaker(redis_client, failure_threshold=1, cooldown_seconds=10)

    await breaker.record_failure("gpt-4o")
    current_time[0] += 11

    first = await breaker.is_open("gpt-4o")
    second = await breaker.is_open("gpt-4o")

    assert first is False  # first caller gets the single trial
    assert second is True  # second caller is blocked — a trial is already in flight
    snap = await breaker.snapshot("gpt-4o")
    assert snap.state == "half_open"
