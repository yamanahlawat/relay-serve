from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.dependencies import get_db_session
from app.model.service import LLMModelService


async def get_model_service(db: AsyncSession = Depends(get_db_session)) -> LLMModelService:
    """
    Get the LLM model service instance with database dependency.
    """
    return LLMModelService(db=db)
