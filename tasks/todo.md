# Day 2 — Provider Adapter (OpenAI only)

Building from scratch (no existing repo in this environment). Reconstructed
the Day 1 foundation minimally, then Day 2 adapter on top.

## Scope change (post Day 2)
Project narrowed from OpenAI + Anthropic to **OpenAI only**. Anthropic
adapter and its tests were deleted rather than kept disabled. Consequences:
- `ChatCompletionRequest.model` is now a bare OpenAI model name (e.g.
  `gpt-4o-mini`) — the `"<provider>/<model>"` prefix convention is gone,
  since there's only one provider.
- "Fallback" (core feature #3 in the original goal) is now **model-to-model
  within OpenAI** (e.g. `gpt-4o` → `gpt-4o-mini` on failure), not
  cross-provider failover. Smaller story for interviews than the original
  plan, but still a real reliability feature — worth being upfront about
  this if it comes up.
- `ProviderAdapter` stays an ABC (not collapsed into `OpenAIAdapter`) so a
  second provider could be added later without reshaping the router or the
  schema, even though only one implementation exists right now.

## Foundation (Day 1, reconstructed)
- [x] app/core/config.py — pydantic-settings, OpenAI creds + timeouts, plus
      circuit-breaker/rate-limit fields (unused until Day 5/7)
- [x] app/schemas/chat.py — ChatCompletionRequest/Response, Message, Usage,
      Choice, GatewayErrorResponse
- [x] app/adapters/base.py — ProviderAdapter ABC, ProviderError(retryable)
- [x] app/main.py — FastAPI app + /health

## Day 2 — Adapter
- [x] app/adapters/openai.py — OpenAIAdapter.chat_completion()
- [x] Map HTTP failures → ProviderError(retryable=...)
- [x] tests/test_openai_adapter.py — respx-mocked success + error paths,
      fallback_models never forwarded upstream
- [x] requirements.txt, .env.example (Anthropic vars removed)
- [x] Run pytest, confirm green

## Design decisions worth flagging
- `retryable` is decided once, at the adapter boundary, based on HTTP
  status / exception type — so Day 5's retry/circuit-breaker layer never
  parses provider-specific error bodies.
- A malformed 2xx response (unexpected JSON shape) is treated as
  `retryable=False` — that's a code/API-contract bug, not a transient
  failure, and retrying won't fix it.
- `fallback_models` is a gateway-only field on the request schema; the
  adapter never forwards it to OpenAI (tested explicitly).

## Review
- `pytest -v`: 6/6 passed (success path, retryable 429/timeout,
  non-retryable 401, malformed-2xx, fallback_models not leaked upstream).
- Live smoke test outside pytest: booted `app.main` via `TestClient`,
  `/health` → 200. Called `OpenAIAdapter.chat_completion()` directly
  against a `respx`-mocked upstream — confirmed request/response mapping
  executes end-to-end, not just inside the test harness.
- Not yet done: streaming, retry/backoff (Day 5), circuit breaker (Day 5),
  router with model-to-model fallback (Day 4).
