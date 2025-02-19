from typing import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas.error import ErrorResponseModel
from app.providers.dependencies import get_model_service
from app.providers.exceptions import DuplicateModelException, ModelNotFoundException, ProviderNotFoundException
from app.providers.models import LLMModel
from app.providers.schemas import ModelCreate, ModelRead, ModelUpdate
from app.providers.schemas.model import ModelsByProvider
from app.providers.services import LLMModelService

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
    service: LLMModelService = Depends(get_model_service),
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
    try:
        return await service.create_model(model_in=model_in)
    except ProviderNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except DuplicateModelException as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.message,
        )


@router.get("/all/", response_model=ModelsByProvider)
async def list_models_by_provider(
    service: LLMModelService = Depends(get_model_service),
) -> dict[str, list[LLMModel]]:
    """
    Retrieves all models across all providers in a single request
    """
    return await service.list_all_models()


@router.get(
    "/",
    response_model=list[ModelRead],
    responses={status.HTTP_200_OK: {"description": "Successfully retrieved list of models", "model": list[ModelRead]}},
)
async def list_models(
    service: LLMModelService = Depends(get_model_service),
    provider_id: UUID | None = None,
    is_active: bool | None = None,
    offset: int = 0,
    limit: int = 10,
) -> Sequence[LLMModel]:
    """
    ## List All LLM Models

    Retrieves a paginated list of all language models, with optional filtering by provider.

    ### Parameters
    - **provider_id** (optional): Filter models by provider UUID
    - **is_active** (optional): Filter by active status
    - **offset** (optional): Number of records to skip (default: 0)
    - **limit** (optional): Maximum number of records to return (default: 10)

    ### Returns
    List of model configurations with their details
    """
    return await service.list_models(provider_id=provider_id, is_active=is_active, offset=offset, limit=limit)


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
    llm_model_id: UUID,
    service: LLMModelService = Depends(get_model_service),
) -> LLMModel:
    """
    ## Get a Specific LLM Model

    Retrieves detailed information about a specific language model by its ID.

    ### Parameters
    - **llm_model_id**: UUID of the model to retrieve

    ### Returns
    Detailed model configuration information
    """
    try:
        return await service.get_model(llm_model_id=llm_model_id)
    except ModelNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )


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
    llm_model_id: UUID,
    service: LLMModelService = Depends(get_model_service),
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
    try:
        return await service.update_model(llm_model_id=llm_model_id, model_in=model_in)
    except ModelNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except DuplicateModelException as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.message,
        )


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
    llm_model_id: UUID,
    service: LLMModelService = Depends(get_model_service),
) -> None:
    """
    ## Delete a Specific LLM Model Configuration

    Permanently removes a language model configuration from the system.

    ### Parameters
    - **llm_model_id**: UUID of the model to delete

    ### Returns
    No content on successful deletion
    """
    try:
        await service.delete_model(llm_model_id=llm_model_id)
    except ModelNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
