from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.providers.crud import crud_model
from app.providers.models import LLMModel


async def check_existing_model(llm_model_id: UUID, db: AsyncSession = Depends(get_db_session)) -> LLMModel:
    """
    Dependency to validate and retrieve an existing model.
    Args:
        llm_model_id: UUID of the model to retrieve
        db: Database session
    Returns:
        The model if it exists
    Raises:
        HTTPException: If the model doesn't exist
    """
    model = await crud_model.get(db=db, id=llm_model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {llm_model_id} not found",
        )
    return model
