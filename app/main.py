from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import keys

app = FastAPI(title="LLM Gateway")
app.include_router(keys.router)


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}
