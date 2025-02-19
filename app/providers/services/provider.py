from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.crud import crud_provider
from app.providers.exceptions import DuplicateProviderException, ProviderNotFoundException
from app.providers.models import LLMProvider
from app.providers.schemas import ProviderCreate, ProviderUpdate


class LLMProviderService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def check_duplicate_name(self, provider_name: str) -> None:
        # Check if provider with same name already exists
        filters = [crud_provider.model.name == provider_name]
        existing_provider = await crud_provider.filter(db=self.db, filters=filters)
        if existing_provider:
            raise DuplicateProviderException(name=provider_name)

    async def create_provider(self, provider_in: ProviderCreate) -> LLMProvider:
        await self.check_duplicate_name(provider_in.name)
        return await crud_provider.create(db=self.db, obj_in=provider_in)

    async def list_providers(
        self, is_active: bool | None = None, offset: int = 0, limit: int = 10
    ) -> Sequence[LLMProvider]:
        filters = []
        if is_active is not None:
            filters.append(crud_provider.model.is_active == is_active)
        return await crud_provider.filter(db=self.db, filters=filters, offset=offset, limit=limit)

    async def get_provider(self, provider_id: UUID) -> LLMProvider:
        provider = await crud_provider.get(db=self.db, id=provider_id)
        if not provider:
            raise ProviderNotFoundException(provider_id=provider_id)
        return provider

    async def update_provider(self, provider_id: UUID, provider_in: ProviderUpdate) -> LLMProvider | None:
        provider = await self.get_provider(provider_id=provider_id)
        if provider_in.name and provider_in.name != provider.name:
            await self.check_duplicate_name(provider_in.name)
        return await crud_provider.update(db=self.db, id=provider.id, obj_in=provider_in)

    async def delete_provider(self, provider_id: UUID):
        provider = await self.get_provider(provider_id=provider_id)
        await crud_provider.delete(db=self.db, id=provider.id)
