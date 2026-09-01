from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.request_logging import RequestLoggingMiddleware
from app.routers import chat, keys

configure_logging(get_settings())

app = FastAPI(title="LLM Gateway")
app.add_middleware(RequestLoggingMiddleware)
app.include_router(keys.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}
