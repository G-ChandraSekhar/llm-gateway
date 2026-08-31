from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    """Raised by an adapter when a provider call fails.

    `retryable` is decided once, here, at the adapter boundary — based on
    the provider's HTTP status code or the kind of transport exception —
    so the retry layer and circuit breaker (Day 5) never have to parse
    provider-specific error bodies to decide whether to retry.
    """

    def __init__(self, message: str, *, provider: str, status_code: int, retryable: bool):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable

    def __repr__(self) -> str:
        return (
            f"ProviderError(provider={self.provider!r}, status_code={self.status_code}, "
            f"retryable={self.retryable}, message={self.message!r})"
        )


class ProviderAdapter(ABC):
    """One implementation per upstream LLM provider. Translates the unified
    gateway schema to/from the provider's native wire format.

    Currently only OpenAIAdapter exists — the project is OpenAI-only.
    `model` on ChatCompletionRequest is just a native OpenAI model name
    (e.g. "gpt-4o-mini"), no provider prefix. This base class is kept
    abstract rather than collapsed into OpenAIAdapter so a second provider
    can be added later without reshaping the router or the schema.
    """

    #: Short provider id, e.g. "openai". Written to ChatCompletionResponse.provider.
    name: str

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Perform a non-streaming chat completion against the provider.

        Must raise ProviderError on any failure (HTTP error, timeout,
        malformed response) — never let a bare httpx exception escape.
        """
        raise NotImplementedError
