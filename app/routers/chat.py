from __future__ import annotations

import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import ProviderError
from app.adapters.openai import OpenAIAdapter
from app.core.adapters import get_openai_adapter
from app.core.auth import get_current_api_key
from app.core.budget import enforce_budget, record_cost_or_warn, record_spend
from app.core.circuit_breaker import CircuitBreaker, get_circuit_breaker
from app.core.config import Settings, get_settings
from app.core.db import get_db
from app.core.rate_limiter import enforce_rate_limit
from app.core.resilience import CircuitOpenError, call_model, call_model_stream
from app.models.api_key import APIKey
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_budget)],
)


def _build_attempt(model: str, status_code: int, retryable: bool, message: str) -> dict:
    return {"model": model, "status_code": status_code, "retryable": retryable, "message": message}


async def _establish_stream(
    body: ChatCompletionRequest,
    adapter: OpenAIAdapter,
    circuit_breaker: CircuitBreaker,
    settings: Settings,
) -> tuple[str, ChatCompletionChunk | None, AsyncIterator[ChatCompletionChunk]]:
    """Fallback loop for the STREAMING path — tries each model until one
    successfully starts (yields a first chunk, or a clean empty stream).
    This is the streaming equivalent of the non-streaming loop below, but
    kept separate because it returns a generator to keep consuming, not a
    single response object.

    Raises HTTPException(502) if every model fails before any content was
    streamed — same shape as the non-streaming endpoint's total-failure
    response, and crucially BEFORE any StreamingResponse has been
    constructed, so the client still gets a real 502 status code rather
    than an error buried inside a 200 SSE stream.
    """
    models_to_try = [body.model, *(body.fallback_models or [])]
    attempts: list[dict] = []

    for model in models_to_try:
        attempt_request = body.model_copy(update={"model": model})
        stream = call_model_stream(adapter, circuit_breaker, attempt_request, settings)
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration:
            # Provider returned a genuinely empty stream — not an error,
            # nothing to fall back from either. This model "won."
            return model, None, stream
        except CircuitOpenError:
            attempts.append(
                _build_attempt(model, 503, True, "circuit open — skipped without calling the provider")
            )
            logger.warning("stream_model_skipped_circuit_open", model=model)
            continue
        except ProviderError as exc:
            attempts.append(_build_attempt(model, exc.status_code, exc.retryable, exc.message))
            logger.warning("stream_model_failed_falling_back", model=model, status_code=exc.status_code)
            continue

        return model, first_chunk, stream

    logger.error("stream_all_models_failed", attempts=attempts)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"error": "All models failed", "attempts": attempts},
    )


def _sse_chunk(chunk: ChatCompletionChunk) -> bytes:
    return f"data: {chunk.model_dump_json()}\n\n".encode()


def _sse_error(exc: ProviderError) -> bytes:
    payload = {"error": {"message": exc.message, "status_code": exc.status_code}}
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _sse_body(
    model: str,
    first_chunk: ChatCompletionChunk | None,
    stream: AsyncIterator[ChatCompletionChunk],
    api_key: APIKey,
    db: AsyncSession,
) -> AsyncIterator[bytes]:
    """Forwards chunks as Server-Sent Events. Once this generator starts
    (i.e. the first chunk has already been established by
    _establish_stream), NO fallback happens on a later failure — a
    mid-stream error becomes a single error event that ends the stream,
    per the "no silent fallback mid-stream" decision. Spend is recorded
    from whichever chunk carries real usage numbers (the final one, if
    OpenAI sent it), after the stream completes successfully.
    """
    usage_prompt_tokens = usage_completion_tokens = None

    try:
        if first_chunk is not None:
            yield _sse_chunk(first_chunk)
            if first_chunk.usage is not None:
                usage_prompt_tokens = first_chunk.usage.prompt_tokens
                usage_completion_tokens = first_chunk.usage.completion_tokens
        async for chunk in stream:
            yield _sse_chunk(chunk)
            if chunk.usage is not None:
                usage_prompt_tokens = chunk.usage.prompt_tokens
                usage_completion_tokens = chunk.usage.completion_tokens
    except ProviderError as exc:
        yield _sse_error(exc)
        yield b"data: [DONE]\n\n"
        return

    if usage_prompt_tokens is not None:
        cost_micros = record_cost_or_warn(model, usage_prompt_tokens, usage_completion_tokens)
        if cost_micros is not None:
            await record_spend(db, api_key.id, cost_micros)

    yield b"data: [DONE]\n\n"


# response_model is intentionally omitted: this endpoint returns EITHER a
# plain ChatCompletionResponse (non-streaming) OR a StreamingResponse
# (streaming) depending on body.stream, and FastAPI's response_model
# validation only works for one fixed shape. Both branches are still
# fully typed at the code level; this just opts out of the automatic
# OpenAPI/response validation for this one route.
@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
    adapter: OpenAIAdapter = Depends(get_openai_adapter),
    circuit_breaker: CircuitBreaker = Depends(get_circuit_breaker),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> ChatCompletionResponse | StreamingResponse:
    """Auth, rate limiting (Day 7), and budget enforcement (Day 8) all run
    once per incoming request, before any model is attempted — a
    rate-limited or over-budget request gets a 429/402 immediately, since
    every model would hit the identical per-key limit; trying a different
    model can't help the way it can for a provider failure. This applies
    identically whether or not `body.stream` is set.

    Tries `body.model` first, then each model in `body.fallback_models`
    in order, on ANY failure — not just retryable ones (Day 4's decision:
    an invalid-request error against gpt-4o might still succeed against
    gpt-4o-mini).

    STREAMING (`body.stream = True`): fallback/retry is only allowed
    while establishing the stream — before any chunk has reached the
    caller. Once a model's stream has genuinely started, the gateway is
    committed to it; a failure after that point ends the stream with a
    single error event, not a silent switch to another model, since the
    client may already be rendering partial content it can't un-receive.

    On success, spend is recorded using the REAL post-call token usage
    from whichever model actually served the request (not the pricing of
    `body.model` if a fallback ended up serving it instead) — see
    app/core/pricing.py and app/core/budget.py.

    If every model fails, the caller gets a 502 listing what was tried and
    why each one failed — not just the last error as if fallback and
    retry never happened.
    """
    if body.stream:
        model, first_chunk, stream = await _establish_stream(body, adapter, circuit_breaker, settings)
        return StreamingResponse(
            _sse_body(model, first_chunk, stream, api_key, db),
            media_type="text/event-stream",
        )

    models_to_try = [body.model, *(body.fallback_models or [])]
    attempts: list[dict] = []

    for model in models_to_try:
        attempt_request = body.model_copy(update={"model": model})
        try:
            response = await call_model(adapter, circuit_breaker, attempt_request, settings)
        except CircuitOpenError:
            attempts.append(
                _build_attempt(model, 503, True, "circuit open — skipped without calling the provider")
            )
            logger.warning("model_skipped_circuit_open", model=model)
            continue
        except ProviderError as exc:
            attempts.append(_build_attempt(model, exc.status_code, exc.retryable, exc.message))
            logger.warning("model_failed_falling_back", model=model, status_code=exc.status_code)
            continue

        cost_micros = record_cost_or_warn(
            response.model, response.usage.prompt_tokens, response.usage.completion_tokens
        )
        if cost_micros is not None:
            await record_spend(db, api_key.id, cost_micros)
        return response

    logger.error("all_models_failed", attempts=attempts)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"error": "All models failed", "attempts": attempts},
    )
