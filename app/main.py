from fastapi import FastAPI

from app.core.config import get_settings

app = FastAPI(title="LLM Gateway")


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}
