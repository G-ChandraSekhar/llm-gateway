from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from threading import Lock

from app.core.config import get_settings


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Circuit:
    failure_count: int = 0
    state: CircuitState = CircuitState.CLOSED
    opened_at: float | None = None


class CircuitBreaker:
    """Tracks failures per key (here, per model name) and opens that key's
    circuit after `failure_threshold` consecutive failures, so a model
    that's clearly down stops being called — including retries — until a
    cooldown elapses.

    In-memory and per-process: state resets on restart and isn't shared
    across gateway instances. That's a deliberate, documented limitation
    for now — Day 7 swaps the backing store for Redis so state is shared,
    without changing this class's public interface (is_open /
    record_success / record_failure).
    """

    def __init__(self, failure_threshold: int, cooldown_seconds: float, clock=time.monotonic):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}
        self._lock = Lock()

    def _get(self, key: str) -> _Circuit:
        if key not in self._circuits:
            self._circuits[key] = _Circuit()
        return self._circuits[key]

    def is_open(self, key: str) -> bool:
        """True if a call for this key should be skipped right now."""
        with self._lock:
            circuit = self._get(key)
            if circuit.state != CircuitState.OPEN:
                return False

            assert circuit.opened_at is not None
            if self._clock() - circuit.opened_at >= self._cooldown_seconds:
                # Cooldown elapsed: let exactly one trial call through
                # before deciding whether to close or reopen.
                circuit.state = CircuitState.HALF_OPEN
                return False
            return True

    def record_success(self, key: str) -> None:
        with self._lock:
            circuit = self._get(key)
            circuit.failure_count = 0
            circuit.state = CircuitState.CLOSED
            circuit.opened_at = None

    def record_failure(self, key: str) -> None:
        with self._lock:
            circuit = self._get(key)
            if circuit.state == CircuitState.HALF_OPEN:
                # The trial call failed — reopen immediately and restart
                # the cooldown, rather than counting toward the threshold
                # again from scratch.
                circuit.state = CircuitState.OPEN
                circuit.opened_at = self._clock()
                return

            circuit.failure_count += 1
            if circuit.failure_count >= self._failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = self._clock()

    def snapshot(self, key: str) -> dict:
        """Read-only view of a circuit's current state — for tests and
        future observability endpoints, not used in the request path.
        """
        with self._lock:
            circuit = self._get(key)
            return {"state": circuit.state.value, "failure_count": circuit.failure_count}


@lru_cache
def get_circuit_breaker() -> CircuitBreaker:
    settings = get_settings()
    return CircuitBreaker(settings.circuit_breaker_failure_threshold, settings.circuit_breaker_cooldown_seconds)
