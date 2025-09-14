"""Provider API router."""

from typing import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas.error import ErrorResponseModel
from app.provider.dependencies import get_provider_service
from app.provider.exceptions import DuplicateProviderException, ProviderNotFoundException
from app.provider.model import LLMProvider
from app.provider.schema import ProviderCreate, ProviderRead, ProviderUpdate
from app.provider.service import LLMProviderService

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
    service: LLMProviderService = Depends(get_provider_service),
) -> LLMProvider:
    """
    ## Create a New LLM Provider

    Creates a new language model provider configuration. Each provider must have a unique name.

    ### Parameters
    - **provider_in**: Provider creation parameters containing:
    - **name**: Name of the provider (must be unique)
    - **type**: Type of provider (openai, anthropic, etc.)
    - **api_key**: API key for the provider (optional)
    - **base_url**: Custom base URL (optional)
    - **is_active**: Whether the provider is active (default: True)

    ### Returns
    The created provider configuration
    """
    try:
        return await service.create_provider(provider_in=provider_in)
    except DuplicateProviderException as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.message,
        )


@router.get(
    "/",
    response_model=list[ProviderRead],
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved list of providers", "model": list[ProviderRead]}
    },
)
async def list_providers(
    service: LLMProviderService = Depends(get_provider_service),
    is_active: bool | None = None,
    provider_name: str | None = None,
    offset: int = 0,
    limit: int = 10,
) -> Sequence[LLMProvider]:
    """
    ## List All LLM Providers

    Retrieves a paginated list of all language model providers.

    ### Parameters
    - **is_active** (optional): Filter by active status
    - **provider_name** (optional): Filter providers by name
    - **offset** (optional): Number of records to skip (default: 0)
    - **limit** (optional): Maximum number of records to return (default: 10)

    ### Returns
    List of provider configurations with their details
    """
    return await service.list_providers(
        is_active=is_active,
        provider_name=provider_name,
        offset=offset,
        limit=limit,
    )


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
    provider_id: UUID,
    service: LLMProviderService = Depends(get_provider_service),
) -> LLMProvider:
    """
    ## Get a Specific LLM Provider

    Retrieves detailed information about a specific language model provider by its ID.

    ### Parameters
    - **provider_id**: UUID of the provider to retrieve

    ### Returns
    Detailed provider configuration information
    """
    try:
        return await service.get_provider(provider_id=provider_id)
    except ProviderNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )


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
    provider_id: UUID,
    service: LLMProviderService = Depends(get_provider_service),
) -> LLMProvider | None:
    """
    ## Update a Specific LLM Provider

    Updates the configuration of an existing language model provider.

    ### Parameters
    - **provider_id**: UUID of the provider to update
    - **provider_in**: Provider update parameters containing:
    - **name** (optional): New name for the provider
    - **type** (optional): Updated provider type
    - **api_key** (optional): Updated API key
    - **base_url** (optional): Updated base URL
    - **is_active** (optional): Updated active status

    ### Returns
    Updated provider configuration
    """
    try:
        return await service.update_provider(provider_id=provider_id, provider_in=provider_in)
    except ProviderNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except DuplicateProviderException as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.message,
        )


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
    provider_id: UUID,
    service: LLMProviderService = Depends(get_provider_service),
) -> None:
    """
    ## Delete a Specific LLM Provider

    Permanently removes a language model provider configuration from the system.

    ### Parameters
    - **provider_id**: UUID of the provider to delete

    ### Returns
    No content on successful deletion
    """
    try:
        await service.delete_provider(provider_id=provider_id)
    except ProviderNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
