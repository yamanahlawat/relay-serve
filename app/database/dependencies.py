from typing import AsyncGenerator

from app.database.session import AsyncSessionLocal


async def get_db_session() -> AsyncGenerator:
    """
    Returns an async generator object that provides an asynchronous database session.
    """
    async with AsyncSessionLocal() as database_session:
        yield database_session
