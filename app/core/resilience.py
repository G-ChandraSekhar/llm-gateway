from __future__ import annotations

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.adapters.base import ProviderAdapter, ProviderError
from app.core.circuit_breaker import CircuitBreaker
from app.core.config import Settings
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


class CircuitOpenError(Exception):
    """Raised when a model's circuit breaker is open. Distinct from
    ProviderError because no request was actually sent to the provider —
    the call was skipped entirely.
    """

    def __init__(self, model: str):
        super().__init__(f"circuit open for model={model!r}")
        self.model = model


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


async def call_model(
    adapter: ProviderAdapter,
    circuit_breaker: CircuitBreaker,
    request: ChatCompletionRequest,
    settings: Settings,
) -> ChatCompletionResponse:
    """Calls one model with retry (same model, retryable failures only,
    exponential backoff + full jitter) guarded by that model's circuit
    breaker.

    A non-retryable failure (e.g. 400, or a malformed response) is NOT
    retried — it raises on the first attempt, since retrying an identical
    bad request can't succeed and only delays falling back to a different
    model (Day 4's job, one layer up from this one).

    Raises CircuitOpenError (no network call made) if the circuit is
    open, or the underlying ProviderError if every retry attempt failed.
    """
    if await circuit_breaker.is_open(request.model):
        raise CircuitOpenError(request.model)

    retryer = AsyncRetrying(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_random_exponential(
            multiplier=settings.retry_base_delay_seconds, max=settings.retry_max_delay_seconds
        ),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )

    try:
        response = await retryer(adapter.chat_completion, request)
    except ProviderError:
        await circuit_breaker.record_failure(request.model)
        raise
    else:
        await circuit_breaker.record_success(request.model)
        return response
