from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    """Unified request schema. OpenAI-only project, so `model` is just a
    native OpenAI model name (e.g. "gpt-4o-mini") — no provider prefix.
    """

    model: str
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False

    # Gateway-only field, never forwarded to OpenAI. Ordered list of
    # alternate OpenAI model names to try if `model` fails or its circuit
    # is open — e.g. ["gpt-4o-mini"] as a fallback for "gpt-4o". This is
    # model-to-model fallback within OpenAI, not cross-provider.
    fallback_models: Optional[list[str]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    provider: str
    choices: list[Choice]
    usage: Usage


class GatewayErrorResponse(BaseModel):
    error: str
    provider: Optional[str] = None
    retryable: bool = False
    status_code: int = 500
