from app.core.circuit_breaker import CircuitBreaker, CircuitState


class FakeClock:
    def __init__(self, start: float = 0.0):
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)

    assert cb.is_open("gpt-4o") is False
    assert cb.snapshot("gpt-4o")["state"] == CircuitState.CLOSED.value


def test_opens_after_threshold_consecutive_failures():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)

    cb.record_failure("gpt-4o")
    cb.record_failure("gpt-4o")
    assert cb.is_open("gpt-4o") is False  # 2 failures, threshold is 3

    cb.record_failure("gpt-4o")
    assert cb.is_open("gpt-4o") is True


def test_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10)

    cb.record_failure("gpt-4o")
    cb.record_failure("gpt-4o")
    cb.record_success("gpt-4o")

    assert cb.snapshot("gpt-4o")["failure_count"] == 0
    cb.record_failure("gpt-4o")
    cb.record_failure("gpt-4o")
    assert cb.is_open("gpt-4o") is False  # only 2 since the reset


def test_different_models_have_independent_circuits():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10)

    cb.record_failure("gpt-4o")
    cb.record_failure("gpt-4o")

    assert cb.is_open("gpt-4o") is True
    assert cb.is_open("gpt-4o-mini") is False


def test_stays_open_until_cooldown_elapses():
    clock = FakeClock(start=0.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)

    cb.record_failure("gpt-4o")
    assert cb.is_open("gpt-4o") is True

    clock.advance(29)
    assert cb.is_open("gpt-4o") is True  # still within cooldown

    clock.advance(2)  # now 31s elapsed, past the 30s cooldown
    assert cb.is_open("gpt-4o") is False


def test_half_open_success_closes_circuit():
    clock = FakeClock(start=0.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, clock=clock)

    cb.record_failure("gpt-4o")
    clock.advance(11)
    assert cb.is_open("gpt-4o") is False  # transitions to half-open, trial allowed

    cb.record_success("gpt-4o")

    assert cb.snapshot("gpt-4o")["state"] == CircuitState.CLOSED.value
    assert cb.is_open("gpt-4o") is False


def test_half_open_failure_reopens_circuit_and_restarts_cooldown():
    clock = FakeClock(start=0.0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, clock=clock)

    cb.record_failure("gpt-4o")
    clock.advance(11)
    assert cb.is_open("gpt-4o") is False  # half-open trial allowed

    cb.record_failure("gpt-4o")  # trial call failed
    assert cb.is_open("gpt-4o") is True

    clock.advance(9)
    assert cb.is_open("gpt-4o") is True  # cooldown restarted, not the original clock

    clock.advance(2)  # 11s since the reopen
    assert cb.is_open("gpt-4o") is False
