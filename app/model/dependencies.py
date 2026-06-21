from typing import Annotated

from fastapi import Depends

from app.core.database.dependencies import DBSessionDep
from app.model.service import LLMModelService


async def get_model_service(db: DBSessionDep) -> LLMModelService:
    """
    Get the LLM model service instance with database dependency.
    """
    return LLMModelService(db=db)


LLMModelServiceDep = Annotated[LLMModelService, Depends(get_model_service)]
