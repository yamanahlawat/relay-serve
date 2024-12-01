from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.providers.services import LLMProviderService


async def get_provider_service(db: AsyncSession = Depends(get_db_session)) -> LLMProviderService:
    return LLMProviderService(db=db)
