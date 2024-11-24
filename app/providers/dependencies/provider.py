from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.providers.crud import crud_provider
from app.providers.models import LLMProvider


async def validate_provider(provider_id: UUID, db: AsyncSession = Depends(get_db_session)) -> LLMProvider:
    """
    Dependency to validate and retrieve an existing provider.
    Args:
        provider_id: UUID of the provider to retrieve
        db: Database session
    Returns:
        The provider if it exists
    Raises:
        HTTPException: If the provider doesn't exist
    """
    provider = await crud_provider.get(db=db, id=provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with id {provider_id} not found",
        )
    return provider
