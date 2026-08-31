from __future__ import annotations

import uuid

import httpx

from app.adapters.base import ProviderAdapter, ProviderError
from app.core.config import Settings
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

# Status codes worth retrying: rate limiting and transient server-side
# failures. 4xx client errors (bad request, auth, not found) are not in
# here — retrying an identical malformed/unauthorized request just wastes
# the retry budget.
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        # Accepts an injected client so tests can mock at the transport
        # layer (respx) without needing a real base_url/timeout dance.
        self._client = client or httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=settings.provider_request_timeout_seconds,
        )

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        try:
            resp = await self._client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._settings.openai_api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"OpenAI request timed out: {exc}",
                provider=self.name,
                status_code=504,
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            # DNS failure, connection refused, connection reset — all worth
            # a retry, distinct from a timeout.
            raise ProviderError(
                f"OpenAI transport error: {exc}",
                provider=self.name,
                status_code=503,
                retryable=True,
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI returned {resp.status_code}: {resp.text[:500]}",
                provider=self.name,
                status_code=resp.status_code,
                retryable=resp.status_code in _RETRYABLE_STATUS_CODES,
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return ChatCompletionResponse(
                id=data.get("id", str(uuid.uuid4())),
                model=data.get("model", request.model),
                provider=self.name,
                choices=[
                    Choice(
                        index=choice.get("index", 0),
                        message=Message(
                            role=choice["message"]["role"],
                            content=choice["message"]["content"] or "",
                        ),
                        finish_reason=choice.get("finish_reason"),
                    )
                ],
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
            )
        except (KeyError, IndexError, TypeError) as exc:
            # A 2xx with a shape we don't recognize is a gateway/API-contract
            # bug, not a transient failure — retrying won't fix it.
            raise ProviderError(
                f"OpenAI returned malformed response: {exc}",
                provider=self.name,
                status_code=502,
                retryable=False,
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
