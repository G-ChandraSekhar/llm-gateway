from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.adapters.base import ProviderError
from app.adapters.openai import OpenAIAdapter
from app.core.adapters import get_openai_adapter
from app.core.auth import get_current_api_key
from app.core.circuit_breaker import CircuitBreaker, get_circuit_breaker
from app.core.config import Settings, get_settings
from app.core.resilience import CircuitOpenError, call_model
from app.models.api_key import APIKey
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
    adapter: OpenAIAdapter = Depends(get_openai_adapter),
    circuit_breaker: CircuitBreaker = Depends(get_circuit_breaker),
    settings: Settings = Depends(get_settings),
) -> ChatCompletionResponse:
    """Tries `body.model` first, then each model in `body.fallback_models`
    in order, on ANY failure — not just retryable ones (Day 4's decision:
    an invalid-request error against gpt-4o might still succeed against
    gpt-4o-mini).

    Within a single model, `call_model` (Day 5) retries on retryable
    failures only — exponential backoff + jitter, same model — and skips
    the call entirely if that model's circuit breaker is open. Moving to
    the next fallback model happens after retries are exhausted (or the
    circuit is open), not instead of them.

    If every model fails, the caller gets a 502 listing what was tried and
    why each one failed — not just the last error as if fallback and
    retry never happened.
    """
    models_to_try = [body.model, *(body.fallback_models or [])]
    attempts: list[dict] = []

    for model in models_to_try:
        attempt_request = body.model_copy(update={"model": model})
        try:
            return await call_model(adapter, circuit_breaker, attempt_request, settings)
        except CircuitOpenError:
            attempts.append(
                {
                    "model": model,
                    "status_code": 503,
                    "retryable": True,
                    "message": "circuit open — too many recent failures, skipped without calling the provider",
                }
            )
        except ProviderError as exc:
            attempts.append(
                {
                    "model": model,
                    "status_code": exc.status_code,
                    "retryable": exc.retryable,
                    "message": exc.message,
                }
            )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"error": "All models failed", "attempts": attempts},
    )
