from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.llms.services import LLMModelService


async def get_model_service(db: AsyncSession = Depends(get_db_session)) -> LLMModelService:
    return LLMModelService(db=db)
