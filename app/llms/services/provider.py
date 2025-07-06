"""Service layer for LLM provider operations."""

from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.llms.crud.provider import crud_provider
from app.llms.exceptions import DuplicateProviderException, ProviderNotFoundException
from app.llms.models.provider import LLMProvider
from app.llms.schemas.provider import ProviderCreate, ProviderUpdate


class LLMProviderService:
    """Service for managing LLM providers."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_provider(self, provider_in: ProviderCreate) -> LLMProvider:
        """
        Create a new LLM provider.
        Args:
            provider_in: Provider creation data
        Returns:
            Created provider
        Raises:
            DuplicateProviderException: If provider name already exists
        """
        # Check if provider name already exists
        existing_provider = await crud_provider.get_by_name(db=self.db, name=provider_in.name)
        if existing_provider:
            raise DuplicateProviderException(provider_name=provider_in.name)

        return await crud_provider.create(db=self.db, obj_in=provider_in)

    async def list_providers(
        self,
        is_active: bool | None = None,
        provider_name: str | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> Sequence[LLMProvider]:
        """
        List providers with optional filtering.
        Args:
            is_active: Filter by active status
            provider_name: Filter by provider name
            offset: Number of records to skip
            limit: Maximum number of records to return
        Returns:
            List of providers
        """
        return await crud_provider.get_multi(
            db=self.db,
            skip=offset,
            limit=limit,
            is_active=is_active,
            name=provider_name,
        )

    async def get_provider(self, provider_id: UUID) -> LLMProvider:
        """
        Get a specific provider by ID.
        Args:
            provider_id: UUID of the provider
        Returns:
            Provider instance
        Raises:
            ProviderNotFoundException: If provider not found
        """
        provider = await crud_provider.get(db=self.db, id=provider_id)
        if not provider:
            raise ProviderNotFoundException(provider_id=str(provider_id))
        return provider

    async def update_provider(self, provider_id: UUID, provider_in: ProviderUpdate) -> LLMProvider:
        """
        Update a provider.
        Args:
            provider_id: UUID of the provider to update
            provider_in: Provider update data
        Returns:
            Updated provider
        Raises:
            ProviderNotFoundException: If provider not found
            DuplicateProviderException: If updated name already exists
        """
        provider = await crud_provider.get(db=self.db, id=provider_id)
        if not provider:
            raise ProviderNotFoundException(provider_id=str(provider_id))

        # Check for duplicate name if name is being updated
        if provider_in.name and provider_in.name != provider.name:
            existing_provider = await crud_provider.get_by_name(db=self.db, name=provider_in.name)
            if existing_provider:
                raise DuplicateProviderException(provider_name=provider_in.name)

        updated_provider = await crud_provider.update(db=self.db, id=provider_id, obj_in=provider_in)
        if not updated_provider:
            raise ProviderNotFoundException(provider_id=str(provider_id))
        return updated_provider

    async def delete_provider(self, provider_id: UUID) -> None:
        """
        Delete a provider.
        Args:
            provider_id: UUID of the provider to delete
        Raises:
            ProviderNotFoundException: If provider not found
        """
        provider = await crud_provider.get(db=self.db, id=provider_id)
        if not provider:
            raise ProviderNotFoundException(provider_id=str(provider_id))

        await crud_provider.delete(db=self.db, id=provider_id)
