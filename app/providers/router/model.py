from typing import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.error import ErrorResponseModel
from app.database.dependencies import get_db_session
from app.providers.crud import crud_model, crud_provider
from app.providers.dependencies import validate_model
from app.providers.models import LLMModel
from app.providers.schemas import ModelCreate, ModelRead, ModelUpdate

router = APIRouter(prefix="/models", tags=["Models"])


@router.post(
    "/",
    response_model=ModelRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Provider not found",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {
                        "Provider not found": {"value": {"detail": "Provider with id {provider_id} not found"}}
                    }
                }
            },
        },
        status.HTTP_409_CONFLICT: {
            "description": "Model already exists",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {"Model exists": {"value": {"detail": "Model {name} already exists for this provider"}}}
                }
            },
        },
    },
)
async def create_model(
    model_in: ModelCreate,
    db: AsyncSession = Depends(get_db_session),
) -> LLMModel:
    """
    ## Create a New LLM Model Configuration

    Creates a new language model configuration for a specific provider. Each model must have a unique name within its provider.

    ### Parameters
    - **model_in**: Model creation parameters containing:
    - **name**: Name of the model (must be unique per provider)
    - **provider_id**: UUID of the provider this model belongs to
    - **config**: Model-specific configuration parameters
    - **is_active**: Whether the model is active (default: True)
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

    return await crud_model.create(db=db, obj_in=model_in)


@router.get(
    "/",
    response_model=list[ModelRead],
    responses={status.HTTP_200_OK: {"description": "Successfully retrieved list of models", "model": list[ModelRead]}},
)
async def list_models(
    db: AsyncSession = Depends(get_db_session),
    provider_id: UUID | None = None,
    offset: int = 0,
    limit: int = 10,
) -> Sequence[LLMModel]:
    """
    ## List All LLM Models

    Retrieves a paginated list of all language models, with optional filtering by provider.

    ### Parameters
    - **provider_id** (optional): Filter models by provider UUID
    - **offset** (optional): Number of records to skip (default: 0)
    - **limit** (optional): Maximum number of records to return (default: 10)

    ### Returns
    List of model configurations with their details
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


@router.get(
    "/{llm_model_id}/",
    response_model=ModelRead,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Model not found",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {"Model not found": {"value": {"detail": "Model with id {llm_model_id} not found"}}}
                }
            },
        }
    },
)
async def get_model(
    llm_model: LLMModel = Depends(validate_model),
) -> LLMModel:
    """
    ## Get a Specific LLM Model

    Retrieves detailed information about a specific language model by its ID.

    ### Parameters
    - **llm_model_id**: UUID of the model to retrieve

    ### Returns
    Detailed model configuration information
    """
    return llm_model


@router.patch(
    "/{llm_model_id}/",
    response_model=ModelRead,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Model not found",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {"Model not found": {"value": {"detail": "Model with id {llm_model_id} not found"}}}
                }
            },
        }
    },
)
async def update_model(
    model_in: ModelUpdate,
    llm_model: LLMModel = Depends(validate_model),
    db: AsyncSession = Depends(get_db_session),
) -> LLMModel | None:
    """
    ## Update a Specific LLM Model Configuration

    Updates the configuration of an existing language model.

    ### Parameters
    - **llm_model_id**: UUID of the model to update
    - **model_in**: Model update parameters containing:
    - **name** (optional): New name for the model
    - **config** (optional): Updated model configuration
    - **is_active** (optional): Updated active status

    ### Returns
    Updated model configuration
    """
    if model_in.name and model_in.name != llm_model.name:
        # Check if model with new name already exists for this provider
        filters = [
            crud_model.model.provider_id == llm_model.provider_id,
            crud_model.model.name == model_in.name,
        ]
        existing_model = await crud_model.filter(db=db, filters=filters)
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model {model_in.name} already exists for this provider",
            )
    return await crud_model.update(db=db, id=llm_model.id, obj_in=model_in)


@router.delete(
    "/{llm_model_id}/",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Model not found",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {"Model not found": {"value": {"detail": "Model with id {llm_model_id} not found"}}}
                }
            },
        }
    },
)
async def delete_model(
    llm_model: LLMModel = Depends(validate_model),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    ## Delete a Specific LLM Model Configuration

    Permanently removes a language model configuration from the system.

    ### Parameters
    - **llm_model_id**: UUID of the model to delete

    ### Returns
    No content on successful deletion
    """
    await crud_model.delete(db=db, id=llm_model.id)
