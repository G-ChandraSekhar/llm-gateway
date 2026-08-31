# LLM Gateway

API proxy in front of OpenAI — one schema, retries, model-to-model
fallback (e.g. gpt-4o → gpt-4o-mini), rate limiting, per-key budgets,
gateway-issued auth.

See `tasks/todo.md` for day-by-day build status and design notes.

## Stack
Python 3.11+ · FastAPI (async) · httpx (OpenAI only) · tenacity · Redis ·
PostgreSQL (SQLAlchemy 2.0 async + asyncpg + Alembic) · structlog ·
pytest + respx · Docker Compose

## Local setup (from scratch)

```bash
# 1. clone / cd into the repo (see "Push to GitHub" below if you haven't made one yet)
cd llm-gateway

# 2. create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. copy env template and fill in provider keys
cp .env.example .env
# edit .env: OPENAI_API_KEY=...

# 5. run the test suite
pytest -v

# 6. boot the app
uvicorn app.main:app --reload
# then in another terminal:
curl http://localhost:8000/health
```

Postgres and Redis aren't needed yet — nothing in the code touches them
until Day 3 (Postgres) and Day 7 (Redis). Day 10 adds a `docker-compose.yml`
that runs both alongside the gateway, so no manual install is planned.

## Push to GitHub (first time)

```bash
git init
git add .
git commit -m "Day 1-2: project scaffold, unified schema, OpenAI adapter"

# create an empty repo on github.com first (no README/gitignore/license —
# we already have them), then:
git remote add origin git@github.com:<your-username>/llm-gateway.git
git branch -M main
git push -u origin main
```

## Project structure
```
llm-gateway/
├── .env.example
├── .gitignore
├── requirements.txt
├── app/
│   ├── main.py                  # FastAPI app + /health
│   ├── core/
│   │   └── config.py             # pydantic-settings, env-driven config
│   ├── schemas/
│   │   └── chat.py               # unified ChatCompletionRequest/Response
│   ├── adapters/
│   │   ├── base.py               # ProviderAdapter interface, ProviderError
│   │   └── openai.py             # OpenAIAdapter
│   ├── routers/                  # (empty — Day 4: model-to-model fallback)
│   └── models/                   # (empty — Day 3: SQLAlchemy models)
└── tests/
    └── test_openai_adapter.py
```
