from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


async def require_admin(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Guards admin-only endpoints (key creation, listing, revocation).

    Fails CLOSED if ADMIN_API_KEY isn't configured: an unset secret means
    "nobody can use this endpoint," not "anyone can." Falling back to open
    access when unconfigured is exactly the footgun that leaves a real
    deployment accidentally wide open without anyone noticing.

    Uses hmac.compare_digest for the comparison, not `==` — a naive string
    comparison short-circuits on the first mismatched byte, which leaks
    timing information an attacker could use to guess the secret one byte
    at a time. Same reasoning as the API key hashing in Day 3, just
    applied to a direct secret comparison instead of a hash lookup.
    """
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled: ADMIN_API_KEY is not configured.",
        )

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing admin credentials")

    provided = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(provided, settings.admin_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")
