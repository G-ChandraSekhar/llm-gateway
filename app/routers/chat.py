from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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
from app.core.resilience import CircuitOpenError, call_model
from app.models.api_key import APIKey
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_budget)],
)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
    adapter: OpenAIAdapter = Depends(get_openai_adapter),
    circuit_breaker: CircuitBreaker = Depends(get_circuit_breaker),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> ChatCompletionResponse:
    """Auth, rate limiting (Day 7), and budget enforcement (Day 8) all run
    once per incoming request, before any model is attempted — a
    rate-limited or over-budget request gets a 429/402 immediately, since
    every model would hit the identical per-key limit; trying a different
    model can't help the way it can for a provider failure.

    Tries `body.model` first, then each model in `body.fallback_models`
    in order, on ANY failure — not just retryable ones (Day 4's decision:
    an invalid-request error against gpt-4o might still succeed against
    gpt-4o-mini).

    Within a single model, `call_model` (Day 5) retries on retryable
    failures only — exponential backoff + jitter, same model — and skips
    the call entirely if that model's circuit breaker is open. Moving to
    the next fallback model happens after retries are exhausted (or the
    circuit is open), not instead of them.

    On success, spend is recorded using the REAL post-call token usage
    from whichever model actually served the request (not the pricing of
    `body.model` if a fallback ended up serving it instead) — see
    app/core/pricing.py and app/core/budget.py.

    If every model fails, the caller gets a 502 listing what was tried and
    why each one failed — not just the last error as if fallback and
    retry never happened.
    """
    models_to_try = [body.model, *(body.fallback_models or [])]
    attempts: list[dict] = []

    for model in models_to_try:
        attempt_request = body.model_copy(update={"model": model})
        try:
            response = await call_model(adapter, circuit_breaker, attempt_request, settings)
        except CircuitOpenError:
            attempts.append(
                {
                    "model": model,
                    "status_code": 503,
                    "retryable": True,
                    "message": "circuit open — too many recent failures, skipped without calling the provider",
                }
            )
            continue
        except ProviderError as exc:
            attempts.append(
                {
                    "model": model,
                    "status_code": exc.status_code,
                    "retryable": exc.retryable,
                    "message": exc.message,
                }
            )
            continue

        cost_micros = record_cost_or_warn(response.model, response.usage.prompt_tokens, response.usage.completion_tokens)
        if cost_micros is not None:
            await record_spend(db, api_key.id, cost_micros)
        return response

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"error": "All models failed", "attempts": attempts},
    )
