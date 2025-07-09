from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.llms.crud import crud_provider
from app.llms.exceptions import DuplicateProviderException, ProviderNotFoundException
from app.llms.models import LLMProvider
from app.llms.schemas import ProviderCreate, ProviderUpdate


class LLMProviderService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def check_duplicate_name(self, provider_name: str) -> None:
        """
        Check if a provider with the same name already exists.
        Args:
            provider_name (str): The name of the provider to check.
        Raises:
            DuplicateProviderException: If a provider with the same name already exists.
        """
        # Check if provider with same name already exists
        filters = [crud_provider.model.name == provider_name]
        existing_provider = await crud_provider.filter(db=self.db, filters=filters)
        if existing_provider:
            raise DuplicateProviderException(provider_name=provider_name)

    async def create_provider(self, provider_in: ProviderCreate) -> LLMProvider:
        """
        Create a new LLM provider.
        Args:
            provider_in (ProviderCreate): The provider creation data.
        Returns:
            LLMProvider: The created provider.
        """
        await self.check_duplicate_name(provider_in.name)
        return await crud_provider.create(db=self.db, obj_in=provider_in)

    async def list_providers(
        self,
        is_active: bool | None = None,
        provider_name: str | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> Sequence[LLMProvider]:
        """
        List all LLM providers with optional filtering.
        Args:
            is_active (bool | None, optional): Filter by active status. Defaults to None.
            provider_name (str | None, optional): Filter by provider name. Defaults to None.
            offset (int, optional): Number of records to skip. Defaults to 0.
            limit (int, optional): Maximum number of records to return. Defaults to 10.
        Returns:
            Sequence[LLMProvider]: List of providers.
        """
        filters = []
        if is_active is not None:
            filters.append(crud_provider.model.is_active == is_active)
        if provider_name:
            filters.append(crud_provider.model.name.ilike(f"%{provider_name}%"))
        return await crud_provider.filter(db=self.db, filters=filters, offset=offset, limit=limit)

    async def get_provider(self, provider_id: UUID) -> LLMProvider:
        """
        Get a specific LLM provider by its ID.
        Args:
            provider_id (UUID): The ID of the provider.
        Raises:
            ProviderNotFoundException: If the provider is not found.
        Returns:
            LLMProvider: The requested provider.
        """
        provider = await crud_provider.get(db=self.db, id=provider_id)
        if not provider:
            raise ProviderNotFoundException(provider_id=provider_id)
        return provider

    async def update_provider(self, provider_id: UUID, provider_in: ProviderUpdate) -> LLMProvider | None:
        """
        Update an existing LLM provider.
        Args:
            provider_id (UUID): The ID of the provider to update.
            provider_in (ProviderUpdate): The provider update data.
        Returns:
            LLMProvider | None: The updated provider or None if not found.
        """
        provider = await self.get_provider(provider_id=provider_id)
        if provider_in.name and provider_in.name != provider.name:
            await self.check_duplicate_name(provider_in.name)
        return await crud_provider.update(db=self.db, id=provider.id, obj_in=provider_in)

    async def delete_provider(self, provider_id: UUID):
        """
        Delete an existing LLM provider.
        Args:
            provider_id (UUID): The ID of the provider to delete.
        """
        provider = await self.get_provider(provider_id=provider_id)
        await crud_provider.delete(db=self.db, id=provider.id)
