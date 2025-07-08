"""Dependencies for LLM services."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.llms.services.model import LLMModelService
from app.llms.services.provider import LLMProviderService


def get_provider_service(db: AsyncSession = Depends(get_db_session)) -> LLMProviderService:
    """Get the LLM provider service."""
    return LLMProviderService(db=db)


def get_model_service(db: AsyncSession = Depends(get_db_session)) -> LLMModelService:
    """Get the LLM model service."""
    return LLMModelService(db=db)
