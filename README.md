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

# 6. (optional, for exploring the API locally without installing Postgres)
#    point the app at a throwaway SQLite file and run the migration:
export DATABASE_URL="sqlite+aiosqlite:///./dev.db"
alembic upgrade head

# 7. boot the app
uvicorn app.main:app --reload
# then in another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/keys -H "Content-Type: application/json" -d '{"name": "my key"}'
curl http://localhost:8000/v1/keys/me -H "Authorization: Bearer <api_key from the previous response>"
```

`/health` needs no database. `/v1/keys` and `/v1/keys/me` do — either point
`DATABASE_URL` at SQLite as above for quick local exploration, or run real
Postgres (Day 10 adds `docker-compose.yml` for that; until then,
`docker run -p 5432:5432 -e POSTGRES_PASSWORD=gateway postgres` plus
`DATABASE_URL=postgresql+asyncpg://postgres:gateway@localhost:5432/postgres`
works too).

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
├── alembic.ini
├── requirements.txt
├── alembic/
│   ├── env.py                    # async migrations, URL from Settings
│   └── versions/
│       └── 0001_create_api_keys.py
├── app/
│   ├── main.py                  # FastAPI app, /health, keys router
│   ├── core/
│   │   ├── config.py             # pydantic-settings, env-driven config
│   │   ├── db.py                 # async engine/session, get_db dependency
│   │   ├── security.py           # API key generation + hashing
│   │   └── auth.py               # get_current_api_key dependency
│   ├── schemas/
│   │   └── chat.py               # unified ChatCompletionRequest/Response
│   ├── models/
│   │   ├── base.py               # SQLAlchemy DeclarativeBase
│   │   └── api_key.py            # APIKey ORM model
│   ├── adapters/
│   │   ├── base.py               # ProviderAdapter interface, ProviderError
│   │   └── openai.py             # OpenAIAdapter
│   └── routers/
│       └── keys.py               # POST /v1/keys, GET /v1/keys/me
└── tests/
    ├── conftest.py                # in-memory SQLite fixtures
    ├── test_openai_adapter.py
    ├── test_security.py
    └── test_keys_router.py
```
