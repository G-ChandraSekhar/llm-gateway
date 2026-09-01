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


class ChunkDelta(BaseModel):
    """Incremental piece of a streamed message — unlike Choice.message,
    both fields are optional since a real chunk only ever carries one or
    the other (role on the first chunk, content on subsequent ones, and
    the final chunk carries neither).
    """

    role: Optional[Role] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """One SSE event's worth of a streamed response — the gateway's own
    shape (matches ChatCompletionResponse's field names/conventions), not
    a passthrough of OpenAI's wire format. `usage` is None on every chunk
    except the final one (requested via stream_options.include_usage),
    which also has an empty `choices` list — that's how OpenAI signals
    "this chunk is usage-only, not more content."
    """

    id: str
    model: str
    provider: str
    choices: list[ChunkChoice]
    usage: Optional[Usage] = None


class GatewayErrorResponse(BaseModel):
    error: str
    provider: Optional[str] = None
    retryable: bool = False
    status_code: int = 500
