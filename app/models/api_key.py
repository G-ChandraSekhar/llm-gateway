from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class APIKey(Base):
    """A gateway-issued API key. Callers authenticate with this; the
    gateway maps it to provider credentials the caller never sees.

    Budget enforcement (Day 8): `spent_micros` is incremented after every
    successful call using the provider's real post-call token usage;
    `budget_limit_micros` is checked BEFORE a request is allowed to call
    any model. See app/core/budget.py and app/core/pricing.py.
    """

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # First 12 chars of the raw key, stored in plaintext so a dashboard can
    # show "sk-gw-ab12cd..." without ever re-deriving or storing the secret.
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    # SHA-256 hex digest of the full raw key, looked up on every request.
    # Deliberately NOT bcrypt/argon2 — see app/core/security.py for why.
    hashed_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Integer MICRO-dollars (millionths of a dollar), not cents. Cents
    # rounded every realistic small request straight to 0 — a 15-prompt /
    # 8-completion gpt-4o-mini call costs ~0.0007 cents, so spend would
    # never move under the original Day 3 schema. Micros give six decimal
    # places of precision as an integer, avoiding float accumulation drift
    # while still being fine-grained enough to actually track real spend.
    # NULL limit means unlimited. The public API (POST /v1/keys,
    # GET /v1/keys/me) exposes these as friendly dollar floats — the
    # micros unit is an internal storage detail, not something callers
    # need to think in.
    budget_limit_micros: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spent_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
