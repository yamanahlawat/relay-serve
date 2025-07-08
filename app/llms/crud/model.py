"""Model CRUD operations."""

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.database.crud import CRUDBase
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider
from app.llms.schemas.model import ModelCreate, ModelUpdate


class CRUDModel(CRUDBase[LLMModel, ModelCreate, ModelUpdate]):
    """CRUD operations for LLM Models."""

    async def get_by_provider_and_name(self, db: AsyncSession, provider_id: UUID, name: str) -> LLMModel | None:
        """
        Get a model by provider ID and name.
        Args:
            db: Database session
            provider_id: UUID of the provider
            name: Model name
        Returns:
            Model instance if found, None otherwise
        """
        query = select(self.model).where(self.model.provider_id == provider_id, self.model.name == name)
        result = await db.scalars(query)
        return result.first()

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 10,
        provider_id: UUID | None = None,
        is_active: bool | None = None,
        name: str | None = None,
    ) -> Sequence[LLMModel]:
        """
        Get multiple models with optional filtering.
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            provider_id: Filter by provider ID
            is_active: Filter by active status
            name: Filter by model name
        Returns:
            List of models
        """
        query = select(self.model).options(joinedload(LLMModel.provider))

        if provider_id is not None:
            query = query.where(self.model.provider_id == provider_id)
        if is_active is not None:
            query = query.where(self.model.is_active == is_active)
        if name is not None:
            query = query.where(self.model.name.ilike(f"%{name}%"))

        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.scalars(query)
        return result.all()

    async def list_models_by_provider(self, db: AsyncSession) -> dict[str, list[LLMModel]]:
        """
        List all models grouped by provider name.
        Args:
            db: Database session
        Returns:
            Dictionary with provider names as keys and their models as values
        """
        # Query that joins models with providers and gets all in one go
        query = (
            select(self.model)
            .options(joinedload(LLMModel.provider))
            .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
            .order_by(LLMProvider.name, LLMModel.name)
        )

        result = await db.scalars(query)
        models = result.all()

        # Group models by provider name
        grouped_models: dict[str, list[LLMModel]] = {}
        for model in models:
            provider_name = model.provider.name
            if provider_name not in grouped_models:
                grouped_models[provider_name] = []
            grouped_models[provider_name].append(model)

        return grouped_models


crud_model = CRUDModel(model=LLMModel)
