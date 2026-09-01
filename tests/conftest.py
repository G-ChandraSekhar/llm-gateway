from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.core.db import get_db
from app.main import app
from app.models.base import Base

# Shared admin secret for any test that needs to hit an admin-gated
# endpoint (POST/GET/DELETE /v1/keys). A single constant here, reused
# across test files, is simpler than each file inventing its own.
TEST_ADMIN_KEY = "test-admin-secret"
ADMIN_HEADERS = {"Authorization": f"Bearer {TEST_ADMIN_KEY}"}


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """A fresh in-memory SQLite DB per test, standing in for Postgres in
    day-to-day test runs. (Real Postgres was verified manually once,
    post-Day-10 — see tasks/todo.md — this fixture is still SQLite for
    every-day test speed, not because Postgres compatibility is unknown.)
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

    def _override_get_settings() -> Settings:
        return Settings(admin_api_key=TEST_ADMIN_KEY)

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_settings] = _override_get_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac._test_db_session = db_session  # type: ignore[attr-defined]
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend():
    return "asyncio"
