from typing import Annotated

from fastapi import Depends

from app.core.database.dependencies import DBSessionDep
from app.provider.service import LLMProviderService


async def get_provider_service(db: DBSessionDep) -> LLMProviderService:
    """
    Get the LLM provider service instance with database dependency.
    """
    return LLMProviderService(db=db)


LLMProviderServiceDep = Annotated[LLMProviderService, Depends(get_provider_service)]
