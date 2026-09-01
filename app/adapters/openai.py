from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import httpx

from app.adapters.base import ProviderAdapter, ProviderError
from app.core.config import Settings
from app.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
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

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            # Without this, OpenAI never sends a usage figure for a
            # streamed response at all — the budget tracker (Day 8)
            # would have no real numbers to record spend from for any
            # streamed call.
            "stream_options": {"include_usage": True},
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._settings.openai_api_key}"},
            ) as resp:
                # Entering the context manager gets the response status
                # and headers WITHOUT reading the body yet — so a
                # rejection (429, 500, bad request, ...) is caught here,
                # before anything has been yielded to our caller. This is
                # the boundary call_model_stream relies on to know retry/
                # fallback is still safe.
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"OpenAI returned {resp.status_code}: {body[:500].decode(errors='replace')}",
                        provider=self.name,
                        status_code=resp.status_code,
                        retryable=resp.status_code in _RETRYABLE_STATUS_CODES,
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        # A single malformed SSE line mid-stream isn't
                        # worth killing the whole stream over — OpenAI's
                        # own framing occasionally includes keep-alive or
                        # comment lines that aren't valid JSON payloads.
                        continue

                    choices = data.get("choices") or []
                    usage_data = data.get("usage")

                    chunk_choices: list[ChunkChoice] = []
                    if choices:
                        raw_choice = choices[0]
                        delta = raw_choice.get("delta") or {}
                        chunk_choices.append(
                            ChunkChoice(
                                index=raw_choice.get("index", 0),
                                delta=ChunkDelta(role=delta.get("role"), content=delta.get("content")),
                                finish_reason=raw_choice.get("finish_reason"),
                            )
                        )

                    yield ChatCompletionChunk(
                        id=data.get("id", str(uuid.uuid4())),
                        model=data.get("model", request.model),
                        provider=self.name,
                        choices=chunk_choices,
                        usage=(
                            Usage(
                                prompt_tokens=usage_data.get("prompt_tokens", 0),
                                completion_tokens=usage_data.get("completion_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0),
                            )
                            if usage_data
                            else None
                        ),
                    )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"OpenAI request timed out: {exc}",
                provider=self.name,
                status_code=504,
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"OpenAI transport error: {exc}",
                provider=self.name,
                status_code=503,
                retryable=True,
            ) from exc
