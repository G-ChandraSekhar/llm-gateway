from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_api_key
from app.core.db import get_db
from app.models.api_key import APIKey

logger = logging.getLogger(__name__)


def is_over_budget(api_key: APIKey) -> bool:
    if api_key.budget_limit_micros is None:
        return False  # unlimited
    return api_key.spent_micros >= api_key.budget_limit_micros


async def enforce_budget(api_key: APIKey = Depends(get_current_api_key)) -> None:
    """FastAPI dependency: pre-check only, once per incoming request (not
    per model attempt — same reasoning as Day 7's rate limiter: budget is
    a property of the key, every model attempt would hit the identical
    limit). Reads `api_key.spent_micros` as of when the key was loaded
    for this request; doesn't re-query, so a burst of concurrent requests
    against a key sitting exactly at its limit could all pass this check
    before any of their spend lands — documented race, not fixed, since
    closing it fully would need locking on every single request even when
    nobody's near their budget.
    """
    if is_over_budget(api_key):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "Budget exceeded",
                "spent_usd": api_key.spent_micros / 1_000_000,
                "budget_limit_usd": (api_key.budget_limit_micros or 0) / 1_000_000,
            },
        )


async def record_spend(db: AsyncSession, api_key_id: str, cost_micros: int) -> None:
    """Atomic increment — `spent_micros = spent_micros + cost_micros` as a
    single SQL UPDATE, not a read-modify-write in Python, so concurrent
    requests against the same key never lose an update to a race.
    """
    await db.execute(
        update(APIKey).where(APIKey.id == api_key_id).values(spent_micros=APIKey.spent_micros + cost_micros)
    )
    await db.commit()


def record_cost_or_warn(model: str, prompt_tokens: int, completion_tokens: int) -> int | None:
    """Thin wrapper around pricing lookup that logs when a model has no
    known price, so unpriced spend isn't silently swallowed without a
    trace anywhere.
    """
    from app.core.pricing import compute_cost_micros

    cost_micros = compute_cost_micros(model, prompt_tokens, completion_tokens)
    if cost_micros is None:
        logger.warning("No pricing entry for model=%r — spend NOT recorded for this call", model)
    return cost_micros
