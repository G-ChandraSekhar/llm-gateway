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

---

# Day 3 — Postgres schema (API keys) + auth middleware

## Built
- [x] app/models/base.py — shared SQLAlchemy DeclarativeBase
- [x] app/models/api_key.py — APIKey model (id, name, prefix, hashed_key,
      is_active, created_at, revoked_at, budget_limit_cents, spent_cents)
- [x] app/core/db.py — async engine/session, get_db dependency
- [x] app/core/security.py — generate_api_key, hash_api_key, key_prefix
- [x] app/core/auth.py — get_current_api_key dependency (Bearer token ->
      hash -> DB lookup -> 401 if missing/invalid/inactive)
- [x] app/routers/keys.py — POST /v1/keys (mint), GET /v1/keys/me (whoami,
      proves the auth dependency works)
- [x] alembic.ini, alembic/env.py (async, URL from Settings not hardcoded),
      alembic/versions/0001_create_api_keys.py
- [x] tests/conftest.py — in-memory SQLite swapped in for Postgres via
      dependency override
- [x] tests/test_security.py, tests/test_keys_router.py

## Design decisions worth flagging
- **Key hashing is SHA-256, not bcrypt/argon2.** The raw key is already a
  256-bit random secret (via `secrets.token_urlsafe`), not a human-chosen
  password — there's no brute-force risk a slow hash would mitigate, and
  this hash runs on every single request, so speed matters here in a way
  it doesn't for login. Documented in the code, not just here.
- **Lookup is by exact hash, not prefix.** `prefix` is stored only for
  display (a dashboard could show "sk-gw-9ao8ci..."); auth always matches
  on the full hash, so no two keys can accidentally collide on lookup.
- **`POST /v1/keys` has no admin auth.** Anyone who can reach the gateway
  can mint a key right now. Flagged explicitly in the endpoint's docstring
  — needs an admin credential or an out-of-band provisioning path before
  this is exposed anywhere. Not fixed yet because Day 3's scope is schema
  + auth *shape*, not admin authorization — but it's a real gap, not an
  oversight.
- **Budget fields exist, enforcement doesn't.** `budget_limit_cents` /
  `spent_cents` are on the schema now; nothing increments or checks them
  until Day 8.

## Known gap — verify against real Postgres
Everything here was tested against SQLite (in-memory for pytest, a real
`.db` file for the Alembic up/down check and the live uvicorn run) — I
have no way to run actual Postgres in this environment. The ORM layer and
migration both use generic SQLAlchemy types that map cleanly to Postgres,
so I'd expect this to just work, but "I'd expect" isn't proof. Before
trusting this in an interview: run `docker run -p 5432:5432 -e
POSTGRES_PASSWORD=gateway postgres`, point `DATABASE_URL` at it, run
`alembic upgrade head`, and hit the same three endpoints (`/health`,
`POST /v1/keys`, `GET /v1/keys/me`) once for real.

## Review
- `pytest -v`: 15/15 passed (6 OpenAI adapter, 4 key security unit tests,
  5 keys-router integration tests against a real FastAPI app + in-memory DB).
- `alembic upgrade head` / `alembic downgrade base` against a real SQLite
  file: both succeed, resulting schema matches the ORM model exactly
  (checked via `sqlite_master`).
- Live smoke test: booted `uvicorn` as an actual subprocess, hit it over
  real HTTP — `/health`, key creation, authenticated `/v1/keys/me`, and
  both 401 cases (missing header, garbage token). All behaved correctly.
- Not yet done: admin auth on key creation, budget enforcement (Day 8),
  rate limiting (Day 7), router/fallback (Day 4).

---

# Day 4 — Router (model-to-model fallback, OpenAI-only)

## Decisions (handed to the user, not made unilaterally)
- **Fallback triggers on ANY failure**, not just `retryable=True` ones —
  including a 400 invalid-request error. Rationale: a parameter rejected
  by gpt-4o might still be accepted by gpt-4o-mini, so gating fallback on
  `retryable` would miss real recoverable cases. (Day 5's retry/backoff,
  which retries the *same* model, is where `retryable` actually matters.)
- **Total failure returns 502 with every attempt's detail** (model tried,
  status code, retryable flag, message) — not just the last error as if
  fallback silently didn't happen. Easier to debug, and honest about what
  the gateway actually tried.
- **`/v1/chat/completions` requires the gateway API key**, same
  `get_current_api_key` dependency built Day 3.

## Built
- [x] app/core/adapters.py — `get_openai_adapter()`, `lru_cache`-backed
      singleton (same pattern as `get_settings`), so one connection pool
      is shared across requests instead of one per call
- [x] app/routers/chat.py — POST /v1/chat/completions: auth -> try
      `model` -> try each `fallback_models` entry in order -> first
      success wins -> 502 with full attempt list if all fail
- [x] Wired into app/main.py
- [x] tests/test_chat_router.py — auth required, success on primary,
      fallback on retryable failure, fallback on non-retryable failure
      (400), total failure returns 502 with both attempts, no
      fallback_models means exactly one attempt

## Review
- `pytest -v`: 21/21 passed (6 adapter, 4 security, 5 keys-router, 6 new
  chat-router tests).
- Live end-to-end test: booted the real `uvicorn` app AND a separate real
  HTTP server (stand-in for OpenAI) as two actual processes talking over
  real sockets — not respx, not ASGITransport. Confirmed: unauthenticated
  request -> 401; authenticated request where `gpt-4o` returns 429 and
  `gpt-4o-mini` succeeds -> 200, response correctly shows
  `model: "gpt-4o-mini"` and the fallback model's content.
- Not yet done: retry/backoff on a single model (Day 5), circuit breaker
  (Day 5), rate limiting (Day 7), budget enforcement (Day 8). Right now a
  single `ProviderError` on a model is treated as final for that model —
  no same-model retry before moving to fallback.

---

# Day 5 — Retry (backoff + jitter) + per-model circuit breaker

## Decisions (yours, not mine)
- **Circuit breaker is per-model**, not global — gpt-4o can be down while
  gpt-4o-mini stays fully available.
- **State is in-memory**, per-process. Resets on restart, not shared
  across gateway instances if you ever run more than one. Day 7 swaps the
  backing store for Redis without changing `CircuitBreaker`'s public
  interface (`is_open` / `record_success` / `record_failure`).
- **Retry settings use the existing config.py defaults** (3 attempts,
  0.5s base delay, 8s max, exponential + full jitter via tenacity's
  `wait_random_exponential`) — unchanged.

## Built
- [x] app/core/circuit_breaker.py — `CircuitBreaker` (closed/open/half-open
      per key), `get_circuit_breaker()` singleton dependency
- [x] app/core/resilience.py — `call_model()`: retries a single model on
      retryable failures only (tenacity, exponential backoff + jitter),
      skips the call entirely via `CircuitOpenError` if that model's
      circuit is open, records success/failure on the breaker
- [x] app/routers/chat.py updated: each model in the fallback chain now
      goes through `call_model` instead of calling the adapter directly —
      retry and circuit-breaker are per-model, fallback-to-next-model
      still happens after retries are exhausted or the circuit is open
- [x] tests/test_circuit_breaker.py — 7 unit tests (state transitions,
      per-key independence, cooldown timing via an injectable fake clock)
- [x] tests/test_resilience.py — 7 tests (retry succeeds, retry exhausts,
      non-retryable skips retry, circuit breaker records correctly, open
      circuit skips the call)
- [x] tests/test_chat_router.py — 2 new integration tests proving retry
      and circuit-open-skip work through the *full* router, not just the
      lower-level resilience tests; existing Day 4 tests' fixture updated
      to use fast/disabled retry settings + an isolated circuit breaker
      per test (the previous fixture would have caused real sleep delays
      and cross-test circuit-breaker pollution once retry logic landed)

## A bug the tests caught before it shipped
The original Day 4 test fixture had no `get_settings`/`get_circuit_breaker`
overrides. Once retry/circuit-breaker logic was wired in, running the old
Day 4 tests against it immediately: (1) added ~3.5s of real wall-clock
sleep to the test run, and (2) broke a test outright — respx ran out of
mocked HTTP responses because a 429 now retries 3 times before falling
back, not once. Both are exactly the kind of thing that's invisible until
you actually run the suite against the new code — fixed by giving each
test isolated, fast settings and a fresh circuit breaker rather than
sharing global state across the test file.

## Review
- `pytest -v`: 37/37 passed in 0.9s (16 new: 7 circuit breaker, 7
  resilience, 2 router-integration).
- Live end-to-end retry test: real `uvicorn` + a real separate fake-OpenAI
  HTTP process. A model failed twice (429) then succeeded on the 3rd real
  call — confirmed via the fake server's own logs showing 3 real network
  hits, all to the *same* model name, with a real ~0.45s delay between the
  1st and 2nd calls (matching the 0.5s base backoff) before the retry.
- Live end-to-end circuit breaker test: same setup, threshold=2. Two real
  failed calls to `gpt-4o-always-down` tripped its circuit; a third
  request (with a fallback model set) resulted in exactly ONE real network
  call — to the fallback model only. The down model's log line never
  appeared for that third request, proving it was skipped without a
  network call, not called-and-failed-again.
- Not yet done: Redis-backed circuit breaker/rate-limit state (Day 7),
  rate limiting itself (Day 7), budget enforcement (Day 8).

---

# Day 7 — Redis-backed rate limiting per API key

## Decisions (yours)
- **Limits both requests/min AND tokens/min**, not just requests.
- **One rate-limit check per incoming request, not per model.** A
  rate-limited key is rejected with 429 immediately — Day 4's fallback
  logic never runs, since every model would hit the identical per-key
  bucket. (This is a real design correction from the initial framing:
  fallback exists for *provider* failures, which differ per model; a rate
  limit is a property of the *key*, which doesn't.)

## Built
- [x] app/core/rate_limiter.py — `RateLimiter`: Redis-backed token
      bucket, atomic via a single Lua script (`EVAL`) covering both the
      requests bucket and tokens bucket in one round trip. If either
      bucket is short, NEITHER is consumed — a request rejected for being
      over the token budget doesn't also waste a unit of the request
      budget.
- [x] app/core/token_estimate.py — `estimate_request_tokens()`: rough
      pre-call estimate (~4 chars/token + `max_tokens` or a default) used
      only for rate-limiting. Explicitly NOT used for billing — Day 8's
      budget tracker will use the provider's real post-call usage numbers,
      and the two are allowed to disagree.
- [x] `enforce_rate_limit` FastAPI dependency, wired into
      `/v1/chat/completions` as a route-level dependency (runs before the
      model-attempt loop, shares the parsed request body with the main
      handler — verified this actually works, not assumed)
- [x] tests/test_rate_limiter.py — 6 tests against `fakeredis` (bucket
      depletion, refill over time via a fake clock, independent per-key
      buckets, and the atomicity guarantee above)
- [x] tests/test_chat_router.py — 2 new integration tests: 429 returned
      with no model ever called even when `fallback_models` is set; two
      different keys have fully independent limits

## Review
- `pytest -v`: 45/45 passed (6 new rate-limiter unit tests, 2 new router
  integration tests).
- Verified the Lua script against a REAL Redis server (not just
  `fakeredis`) — installed `redis-server` directly, ran the same
  `RateLimiter` class against it, inspected the actual Redis hash keys
  afterward (`ratelimit:<key>:requests` / `:tokens`) to confirm the
  stored float levels and timestamps matched the expected math exactly
  (3 req/min bucket, 3 calls succeed, 4th blocked with `retry_after≈20s`,
  matching `(1 - level) / (3/60)`).
- Live end-to-end test: real `uvicorn` + real Redis + a separate real
  fake-OpenAI process, 2 req/min limit. Requests 1-2 succeeded (both
  logged as real network hits by the fake server); request 3 — sent WITH
  a `fallback_models` entry — got a 429 with `retry_after_seconds: 29.93`,
  and the fake server's log shows **zero** entries for that request,
  proving no network call happened at all, fallback included.
- Not yet done: Day 8 budget enforcement (spend tracking against
  `budget_limit_cents`); circuit breaker still in-memory, not yet moved to
  Redis (flagged as a gap since Day 5 — still open, now that Redis is
  actually wired into the project there's no remaining reason not to do
  this next if it matters for the story).

## New local dependency
Redis is now required to run `/v1/chat/completions` for real (not just
`pytest`, which uses `fakeredis`). See README for install instructions.
