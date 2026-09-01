from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

import structlog
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.adapters.base import ProviderAdapter, ProviderError
from app.core.circuit_breaker import CircuitBreaker
from app.core.config import Settings
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse

logger = structlog.get_logger(__name__)


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


def _log_before_sleep(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "retrying_model_call",
        attempt=retry_state.attempt_number,
        status_code=getattr(exc, "status_code", None),
        message=getattr(exc, "message", str(exc)),
    )


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
        before_sleep=_log_before_sleep,
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


async def call_model_stream(
    adapter: ProviderAdapter,
    circuit_breaker: CircuitBreaker,
    request: ChatCompletionRequest,
    settings: Settings,
) -> AsyncIterator[ChatCompletionChunk]:
    """Streaming counterpart to call_model. Retry/circuit-breaker only
    apply to STARTING the stream — pulling the first chunk off the
    adapter's generator, which is also where the adapter surfaces a
    rejected request (bad status code) as a ProviderError. Once that
    first chunk has been yielded successfully, this function stops
    retrying entirely: any later failure from the adapter's generator
    propagates straight to the caller, uncaught, exactly as designed —
    the caller has already committed to this model and can't silently
    switch without corrupting a response the client may have already
    started receiving.

    This is a hand-rolled retry loop, not tenacity like call_model. Not
    a style choice — tenacity's context-manager pattern is for retrying a
    whole call, not "retry only the first item pulled from an async
    generator, then hand back the rest of that same generator untouched."
    There's no clean way to express that with the existing AsyncRetrying
    API, so a plain loop is clearer here than fighting the abstraction.
    """
    if await circuit_breaker.is_open(request.model):
        raise CircuitOpenError(request.model)

    last_error: ProviderError | None = None

    for attempt in range(settings.retry_max_attempts):
        stream = adapter.chat_completion_stream(request)
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration:
            # The provider returned an empty stream — no chunks at all,
            # but also no error. Treat as a (trivial) success rather than
            # retrying or falling back over nothing.
            await circuit_breaker.record_success(request.model)
            return
        except ProviderError as exc:
            last_error = exc
            await circuit_breaker.record_failure(request.model)
            if not exc.retryable or attempt == settings.retry_max_attempts - 1:
                raise
            wait_ceiling = min(
                settings.retry_base_delay_seconds * (2**attempt), settings.retry_max_delay_seconds
            )
            wait_seconds = random.uniform(0, wait_ceiling)
            logger.warning(
                "retrying_stream_start",
                attempt=attempt + 1,
                model=request.model,
                status_code=exc.status_code,
                message=exc.message,
                wait_seconds=round(wait_seconds, 3),
            )
            await asyncio.sleep(wait_seconds)
            continue

        # Stream started successfully — committed to this model now.
        await circuit_breaker.record_success(request.model)
        yield first_chunk
        async for chunk in stream:
            yield chunk
        return

    # Unreachable in practice (the loop always returns or raises above),
    # but keeps type checkers happy and fails loudly instead of silently
    # returning nothing if the loop logic above is ever changed.
    if last_error is not None:
        raise last_error
