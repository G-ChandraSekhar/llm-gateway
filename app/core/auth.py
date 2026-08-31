from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.security import hash_api_key
from app.models.api_key import APIKey

# auto_error=False so a missing header raises our own 401 with a clear
# message, instead of FastAPI's generic "not authenticated".
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> APIKey:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")

    hashed = hash_api_key(credentials.credentials)
    result = await db.execute(select(APIKey).where(APIKey.hashed_key == hashed))
    api_key = result.scalar_one_or_none()

    if api_key is None or not api_key.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive API key")

    return api_key
