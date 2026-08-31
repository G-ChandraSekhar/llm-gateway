from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_api_key
from app.core.db import get_db
from app.core.security import generate_api_key, hash_api_key, key_prefix
from app.models.api_key import APIKey

router = APIRouter(prefix="/v1/keys", tags=["keys"])


class CreateAPIKeyRequest(BaseModel):
    name: str
    budget_limit_cents: int | None = None


class CreateAPIKeyResponse(BaseModel):
    id: str
    name: str
    api_key: str  # the raw key — shown exactly once, never retrievable again
    prefix: str


class APIKeyInfo(BaseModel):
    id: str
    name: str
    prefix: str
    is_active: bool
    budget_limit_cents: int | None
    spent_cents: int


@router.post("", response_model=CreateAPIKeyResponse)
async def create_api_key(
    body: CreateAPIKeyRequest, db: AsyncSession = Depends(get_db)
) -> CreateAPIKeyResponse:
    """KNOWN GAP: this endpoint has no admin auth guarding it — anyone who
    can reach the gateway can mint a key. That's fine for local dev, but
    this MUST be locked behind an admin credential (or replaced with an
    out-of-band provisioning script) before the gateway is exposed
    anywhere. Flagging it here rather than pretending it's already solved.
    """
    raw_key = generate_api_key()
    api_key = APIKey(
        name=body.name,
        prefix=key_prefix(raw_key),
        hashed_key=hash_api_key(raw_key),
        budget_limit_cents=body.budget_limit_cents,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return CreateAPIKeyResponse(id=api_key.id, name=api_key.name, api_key=raw_key, prefix=api_key.prefix)


@router.get("/me", response_model=APIKeyInfo)
async def whoami(api_key: APIKey = Depends(get_current_api_key)) -> APIKeyInfo:
    return APIKeyInfo(
        id=api_key.id,
        name=api_key.name,
        prefix=api_key.prefix,
        is_active=api_key.is_active,
        budget_limit_cents=api_key.budget_limit_cents,
        spent_cents=api_key.spent_cents,
    )
