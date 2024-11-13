from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.providers.crud.provider import crud_provider
from app.providers.schemas.provider import ProviderCreate, ProviderRead, ProviderUpdate

router = APIRouter(prefix="/providers", tags=["Providers"])


@router.post("/", response_model=ProviderRead, status_code=status.HTTP_201_CREATED)
async def create_provider(
    provider_in: ProviderCreate,
    db: AsyncSession = Depends(get_db_session),
) -> ProviderRead:
    """
    Create a new LLM provider.
    """
    # Check if provider with same name already exists
    filters = [crud_provider.model.name == provider_in.name]
    existing_provider = await crud_provider.filter(db=db, filters=filters)
    if existing_provider:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Provider with name {provider_in.name} already exists",
        )

    provider = await crud_provider.create(db=db, obj_in=provider_in)
    return provider


@router.get("/", response_model=list[ProviderRead])
async def list_providers(
    db: AsyncSession = Depends(get_db_session),
    offset: int = 0,
    limit: int = 10,
) -> list[ProviderRead]:
    """
    List all LLM providers.
    """
    providers = await crud_provider.filter(db=db, offset=offset, limit=limit)
    return providers


@router.get("/{provider_id}/", response_model=ProviderRead)
async def get_provider(
    provider_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> ProviderRead:
    """
    Get a specific LLM provider by ID.
    """
    provider = await crud_provider.get(db=db, id=provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with id {provider_id} not found",
        )
    return provider


@router.patch("/{provider_id}/", response_model=ProviderRead)
async def update_provider(
    provider_id: UUID,
    provider_in: ProviderUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> ProviderRead:
    """
    Update a specific LLM provider.
    """
    provider = await crud_provider.get(db=db, id=provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with id {provider_id} not found",
        )

    provider = await crud_provider.update(db=db, id=provider_id, obj_in=provider_in)
    return provider


@router.delete("/{provider_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(
    provider_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    Delete a specific LLM provider.
    """
    provider = await crud_provider.get(db=db, id=provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with id {provider_id} not found",
        )

    await crud_provider.delete(db=db, id=provider_id)
