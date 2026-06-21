from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import AsyncSessionLocal


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Returns an async generator object that provides an asynchronous database session.
    """
    async with AsyncSessionLocal() as database_session:
        yield database_session


# Reusable annotated dependency for an async database session.
DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
