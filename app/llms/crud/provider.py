"""Provider CRUD operations."""

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.crud import CRUDBase
from app.llms.models.provider import LLMProvider
from app.llms.schemas.provider import ProviderCreate, ProviderUpdate


class CRUDProvider(CRUDBase[LLMProvider, ProviderCreate, ProviderUpdate]):
    """CRUD operations for LLM providers."""

    async def get_by_name(self, db: AsyncSession, name: str) -> LLMProvider | None:
        """
        Get a provider by name.
        Args:
            db: Database session
            name: Provider name
        Returns:
            Provider instance if found, None otherwise
        """
        query = select(self.model).where(self.model.name == name)
        result = await db.scalars(query)
        return result.first()

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 10,
        is_active: bool | None = None,
        name: str | None = None,
    ) -> Sequence[LLMProvider]:
        """
        Get multiple providers with optional filtering.
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            is_active: Filter by active status
            name: Filter by provider name
        Returns:
            List of providers
        """
        query = select(self.model)

        if is_active is not None:
            query = query.where(self.model.is_active == is_active)
        if name is not None:
            query = query.where(self.model.name.ilike(f"%{name}%"))

        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.scalars(query)
        return result.all()


crud_provider = CRUDProvider(model=LLMProvider)
