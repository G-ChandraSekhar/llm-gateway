from __future__ import annotations

from app.schemas.chat import ChatCompletionRequest

# ~4 characters per token is a standard rough estimate for English text
# with GPT-style tokenizers. This is deliberately approximate — it exists
# only to keep the rate limiter's pre-call check fast and dependency-free
# (no real tokenizer call). Day 8's budget tracker uses the exact
# post-call usage numbers from OpenAI's response for anything billing-
# related; this number is never used for that.
_CHARS_PER_TOKEN_ESTIMATE = 4
_DEFAULT_COMPLETION_TOKENS_ESTIMATE = 512


def estimate_request_tokens(request: ChatCompletionRequest) -> int:
    char_count = sum(len(m.content) for m in request.messages)
    prompt_estimate = max(char_count // _CHARS_PER_TOKEN_ESTIMATE, 1)
    completion_estimate = request.max_tokens or _DEFAULT_COMPLETION_TOKENS_ESTIMATE
    return prompt_estimate + completion_estimate
