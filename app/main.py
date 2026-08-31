from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import chat, keys

app = FastAPI(title="LLM Gateway")
app.include_router(keys.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}
