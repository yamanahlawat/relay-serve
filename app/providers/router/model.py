from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.providers.crud.model import crud_model
from app.providers.crud.provider import crud_provider
from app.providers.schemas.model import ModelCreate, ModelRead, ModelUpdate

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_in: ModelCreate,
    db: AsyncSession = Depends(get_db_session),
) -> ModelRead:
    """
    Create a new LLM model configuration.
    """
    # Verify provider exists
    provider = await crud_provider.get(db=db, id=model_in.provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with id {model_in.provider_id} not found",
        )

    # Check if model with same name exists for this provider
    filters = [crud_model.model.provider_id == model_in.provider_id, crud_model.model.name == model_in.name]
    existing_model = await crud_model.filter(
        db=db,
        filters=filters,
    )
    if existing_model:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {model_in.name} already exists for this provider",
        )

    model = await crud_model.create(db=db, obj_in=model_in)
    return model


@router.get("/", response_model=List[ModelRead])
async def list_models(
    db: AsyncSession = Depends(get_db_session),
    provider_id: UUID | None = None,
    offset: int = 0,
    limit: int = 10,
) -> List[ModelRead]:
    """
    List all LLM models, optionally filtered by provider.
    """
    if provider_id:
        filters = [crud_model.model.provider_id == provider_id]
        models = await crud_model.filter(
            db=db,
            filters=filters,
            offset=offset,
            limit=limit,
        )
    else:
        models = await crud_model.filter(db=db, offset=offset, limit=limit)
    return models


@router.get("/{llm_model_id}/", response_model=ModelRead)
async def get_model(
    llm_model_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> ModelRead:
    """
    Get a specific LLM model by ID.
    """
    model = await crud_model.get(db=db, id=llm_model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {llm_model_id} not found",
        )
    return model


@router.patch("/{llm_model_id}/", response_model=ModelRead)
async def update_model(
    llm_model_id: UUID,
    model_in: ModelUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> ModelRead:
    """
    Update a specific LLM model configuration.
    """
    model = await crud_model.get(db=db, id=llm_model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {llm_model_id} not found",
        )

    model = await crud_model.update(db=db, id=llm_model_id, obj_in=model_in)
    return model


@router.delete("/{llm_model_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    llm_model_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    Delete a specific LLM model configuration.
    """
    model = await crud_model.get(db=db, id=llm_model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {llm_model_id} not found",
        )

    await crud_model.delete(db=db, id=llm_model_id)
