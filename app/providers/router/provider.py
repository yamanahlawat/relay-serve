from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.error import ErrorResponseModel
from app.database.dependencies import get_db_session
from app.providers.crud import crud_provider
from app.providers.dependencies import check_existing_provider
from app.providers.models import LLMProvider
from app.providers.schemas import ProviderCreate, ProviderRead, ProviderUpdate

router = APIRouter(prefix="/providers", tags=["Providers"])


@router.post(
    "/",
    response_model=ProviderRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "Provider already exists",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {"Provider exists": {"value": {"detail": "Provider with name {name} already exists"}}}
                }
            },
        }
    },
)
async def create_provider(
    provider_in: ProviderCreate,
    db: AsyncSession = Depends(get_db_session),
) -> LLMProvider:
    """
    ## Create a New LLM Provider

    Creates a new language model provider configuration. Each provider must have a unique name.

    ### Parameters
    - **provider_in**: Provider creation parameters containing:
    - **name**: Name of the provider (must be unique)
    - **config**: Provider-specific configuration parameters
    - **is_active**: Whether the provider is active (default: True)

    ### Returns
    The created provider configuration
    """
    # Check if provider with same name already exists
    filters = [crud_provider.model.name == provider_in.name]
    existing_provider = await crud_provider.filter(db=db, filters=filters)
    if existing_provider:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Provider with name {provider_in.name} already exists",
        )

    return await crud_provider.create(db=db, obj_in=provider_in)


@router.get(
    "/",
    response_model=list[ProviderRead],
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved list of providers", "model": list[ProviderRead]}
    },
)
async def list_providers(
    db: AsyncSession = Depends(get_db_session),
    offset: int = 0,
    limit: int = 10,
) -> Sequence[LLMProvider]:
    """
    ## List All LLM Providers

    Retrieves a paginated list of all language model providers.

    ### Parameters
    - **offset** (optional): Number of records to skip (default: 0)
    - **limit** (optional): Maximum number of records to return (default: 10)

    ### Returns
    List of provider configurations with their details
    """
    return await crud_provider.filter(db=db, offset=offset, limit=limit)


@router.get(
    "/{provider_id}/",
    response_model=ProviderRead,
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
        }
    },
)
async def get_provider(
    provider: ProviderRead = Depends(check_existing_provider),
) -> ProviderRead:
    """
    ## Get a Specific LLM Provider

    Retrieves detailed information about a specific language model provider by its ID.

    ### Parameters
    - **provider_id**: UUID of the provider to retrieve

    ### Returns
    Detailed provider configuration information
    """
    return provider


@router.patch(
    "/{provider_id}/",
    response_model=ProviderRead,
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
        }
    },
)
async def update_provider(
    provider_in: ProviderUpdate,
    provider: ProviderRead = Depends(check_existing_provider),
    db: AsyncSession = Depends(get_db_session),
) -> LLMProvider | None:
    """
    ## Update a Specific LLM Provider

    Updates the configuration of an existing language model provider.

    ### Parameters
    - **provider_id**: UUID of the provider to update
    - **provider_in**: Provider update parameters containing:
    - **name** (optional): New name for the provider
    - **config** (optional): Updated provider configuration
    - **is_active** (optional): Updated active status

    ### Returns
    Updated provider configuration
    """
    if provider_in.name and provider_in.name != provider.name:
        # Check if provider with new name already exists
        filters = [crud_provider.model.name == provider_in.name]
        existing_provider = await crud_provider.filter(db=db, filters=filters)
        if existing_provider:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Provider with name {provider_in.name} already exists",
            )
    return await crud_provider.update(db=db, id=provider.id, obj_in=provider_in)


@router.delete(
    "/{provider_id}/",
    status_code=status.HTTP_204_NO_CONTENT,
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
        }
    },
)
async def delete_provider(
    provider: ProviderRead = Depends(check_existing_provider),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    ## Delete a Specific LLM Provider

    Permanently removes a language model provider configuration from the system.

    ### Parameters
    - **provider_id**: UUID of the provider to delete

    ### Returns
    No content on successful deletion
    """
    await crud_provider.delete(db=db, id=provider.id)
