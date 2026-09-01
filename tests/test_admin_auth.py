from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.admin_auth import require_admin
from app.core.config import Settings


@pytest.mark.asyncio
async def test_fails_closed_when_admin_key_unconfigured():
    settings = Settings(admin_api_key="")

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(authorization="Bearer anything", settings=settings)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_fails_closed_even_with_no_header_when_unconfigured():
    # Unconfigured should fail closed regardless of what (if anything) the
    # caller sent — an empty admin secret is never "open to everyone."
    settings = Settings(admin_api_key="")

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(authorization=None, settings=settings)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_missing_header_is_401_when_configured():
    settings = Settings(admin_api_key="real-secret")

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(authorization=None, settings=settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing admin credentials"


@pytest.mark.asyncio
async def test_wrong_secret_is_401():
    settings = Settings(admin_api_key="real-secret")

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(authorization="Bearer wrong-secret", settings=settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid admin credentials"


@pytest.mark.asyncio
async def test_correct_secret_passes():
    settings = Settings(admin_api_key="real-secret")

    await require_admin(authorization="Bearer real-secret", settings=settings)  # should not raise


@pytest.mark.asyncio
async def test_non_bearer_scheme_is_401():
    settings = Settings(admin_api_key="real-secret")

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(authorization="Basic real-secret", settings=settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing admin credentials"
