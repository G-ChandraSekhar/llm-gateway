from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency. Tests override this entirely (swap in a SQLite
    session-maker) rather than pointing `engine` at a different URL, so
    route handlers never need to know which backend they're talking to.
    """
    async with SessionLocal() as session:
        yield session
