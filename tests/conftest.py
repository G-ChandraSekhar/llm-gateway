from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.db import get_db
from app.main import app
from app.models.base import Base


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """A fresh in-memory SQLite DB per test, standing in for Postgres.

    Same SQLAlchemy models and queries run against both — this exercises
    the ORM layer for real, just not against the actual Postgres wire
    protocol. Known gap: this doesn't catch Postgres-specific behavior
    (e.g. server-side defaults evaluated differently, dialect-specific
    types). Worth a one-time manual test against real Postgres before
    treating Day 3 as fully verified.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac._test_db_session = db_session  # type: ignore[attr-defined]
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend():
    return "asyncio"
