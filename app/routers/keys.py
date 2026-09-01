from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.admin_auth import require_admin
from app.core.auth import get_current_api_key
from app.core.db import get_db
from app.core.security import generate_api_key, hash_api_key, key_prefix
from app.models.api_key import APIKey

router = APIRouter(prefix="/v1/keys", tags=["keys"])

_USD_TO_MICROS = 1_000_000


class CreateAPIKeyRequest(BaseModel):
    name: str
    # Friendly dollar amount at the API boundary — stored internally as
    # integer micros (see app/models/api_key.py for why).
    budget_limit_usd: float | None = None


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
    budget_limit_usd: float | None
    spent_usd: float


def _to_info(api_key: APIKey) -> APIKeyInfo:
    return APIKeyInfo(
        id=api_key.id,
        name=api_key.name,
        prefix=api_key.prefix,
        is_active=api_key.is_active,
        budget_limit_usd=(
            api_key.budget_limit_micros / _USD_TO_MICROS if api_key.budget_limit_micros is not None else None
        ),
        spent_usd=api_key.spent_micros / _USD_TO_MICROS,
    )


@router.post("", response_model=CreateAPIKeyResponse, dependencies=[Depends(require_admin)])
async def create_api_key(
    body: CreateAPIKeyRequest, db: AsyncSession = Depends(get_db)
) -> CreateAPIKeyResponse:
    """Requires ADMIN_API_KEY (see app/core/admin_auth.py) — closes the gap
    flagged since Day 3, where anyone who could reach the gateway could
    mint themselves a key.
    """
    raw_key = generate_api_key()
    budget_limit_micros = (
        round(body.budget_limit_usd * _USD_TO_MICROS) if body.budget_limit_usd is not None else None
    )
    api_key = APIKey(
        name=body.name,
        prefix=key_prefix(raw_key),
        hashed_key=hash_api_key(raw_key),
        budget_limit_micros=budget_limit_micros,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return CreateAPIKeyResponse(id=api_key.id, name=api_key.name, api_key=raw_key, prefix=api_key.prefix)


@router.get("", response_model=list[APIKeyInfo], dependencies=[Depends(require_admin)])
async def list_keys(db: AsyncSession = Depends(get_db)) -> list[APIKeyInfo]:
    """Admin-only. Returns every key, active and revoked — an admin needs
    visibility into revoked keys too (e.g. to confirm a revocation took),
    not just the currently-usable ones.
    """
    result = await db.execute(select(APIKey).order_by(APIKey.created_at.desc()))
    return [_to_info(k) for k in result.scalars().all()]


@router.delete("/{key_id}", response_model=APIKeyInfo, dependencies=[Depends(require_admin)])
async def revoke_key(key_id: str, db: AsyncSession = Depends(get_db)) -> APIKeyInfo:
    """Soft-revoke, not a hard delete: sets is_active=False and
    revoked_at, keeps the row (and its spend/name history) intact for
    audit purposes. A revoked key fails auth immediately on its next
    request (get_current_api_key checks is_active) — no cache or TTL
    delay to worry about, since the check hits the DB every time.
    """
    result = await db.execute(select(APIKey).where(APIKey.id == key_id))
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    api_key.revoked_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(api_key)

    return _to_info(api_key)


@router.get("/me", response_model=APIKeyInfo)
async def whoami(api_key: APIKey = Depends(get_current_api_key)) -> APIKeyInfo:
    """NOT admin-gated — this is a caller looking up their own key's info
    using their own key, not an admin operation.
    """
    return _to_info(api_key)
