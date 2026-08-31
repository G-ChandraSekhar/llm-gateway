from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.budget import enforce_budget, is_over_budget, record_spend
from app.models.api_key import APIKey


def make_key(**overrides) -> APIKey:
    defaults = dict(
        name="test",
        prefix="sk-gw-test",
        hashed_key="deadbeef",
        spent_micros=0,
        budget_limit_micros=None,
    )
    defaults.update(overrides)
    return APIKey(**defaults)


def test_unlimited_budget_never_over():
    key = make_key(budget_limit_micros=None, spent_micros=10_000_000)
    assert is_over_budget(key) is False


def test_under_budget_is_not_over():
    key = make_key(budget_limit_micros=1_000_000, spent_micros=999_999)
    assert is_over_budget(key) is False


def test_at_or_over_budget_is_over():
    key = make_key(budget_limit_micros=1_000_000, spent_micros=1_000_000)
    assert is_over_budget(key) is True

    key2 = make_key(budget_limit_micros=1_000_000, spent_micros=1_500_000)
    assert is_over_budget(key2) is True


@pytest.mark.asyncio
async def test_enforce_budget_raises_402_when_over(db_session: AsyncSession):
    key = make_key(budget_limit_micros=1_000_000, spent_micros=2_000_000)

    with pytest.raises(HTTPException) as exc_info:
        await enforce_budget(api_key=key)

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail["spent_usd"] == 2.0
    assert exc_info.value.detail["budget_limit_usd"] == 1.0


@pytest.mark.asyncio
async def test_enforce_budget_passes_when_under(db_session: AsyncSession):
    key = make_key(budget_limit_micros=1_000_000, spent_micros=500_000)
    await enforce_budget(api_key=key)  # should not raise


@pytest.mark.asyncio
async def test_record_spend_increments_atomically(db_session: AsyncSession):
    key = make_key(spent_micros=0)
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    await record_spend(db_session, key.id, cost_micros=500)
    await record_spend(db_session, key.id, cost_micros=250)

    result = await db_session.execute(select(APIKey).where(APIKey.id == key.id))
    refreshed = result.scalar_one()
    assert refreshed.spent_micros == 750  # both increments landed, not one overwriting the other
