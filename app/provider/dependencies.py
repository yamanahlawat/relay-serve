from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.dependencies import get_db_session
from app.provider.service import LLMProviderService


async def get_provider_service(db: AsyncSession = Depends(get_db_session)) -> LLMProviderService:
    """
    Get the LLM provider service instance with database dependency.
    """
    return LLMProviderService(db=db)
