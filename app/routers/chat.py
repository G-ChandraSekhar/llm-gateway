from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.adapters.base import ProviderError
from app.adapters.openai import OpenAIAdapter
from app.core.adapters import get_openai_adapter
from app.core.auth import get_current_api_key
from app.models.api_key import APIKey
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
    adapter: OpenAIAdapter = Depends(get_openai_adapter),
) -> ChatCompletionResponse:
    """Tries `body.model` first, then each model in `body.fallback_models`
    in order, on ANY failure — not just retryable ones. Deliberate choice:
    an invalid-request error against gpt-4o (e.g. an unsupported param)
    might still succeed against gpt-4o-mini, so fallback isn't gated on
    `retryable`. (Day 5's retry/backoff layer, which retries the SAME
    model, is where `retryable` actually matters.)

    If every attempt fails, the caller gets a 502 listing what was tried
    and why each one failed — not just the last error as if fallback
    never happened.
    """
    models_to_try = [body.model, *(body.fallback_models or [])]
    attempts: list[dict] = []

    for model in models_to_try:
        attempt_request = body.model_copy(update={"model": model})
        try:
            return await adapter.chat_completion(attempt_request)
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
