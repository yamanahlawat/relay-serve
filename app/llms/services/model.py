"""Service layer for LLM model operations."""

from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.llms.crud.model import crud_model
from app.llms.crud.provider import crud_provider
from app.llms.exceptions import DuplicateModelException, ModelNotFoundException, ProviderNotFoundException
from app.llms.models.model import LLMModel
from app.llms.schemas.model import ModelCreate, ModelUpdate


class LLMModelService:
    """Service for managing LLM models."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_model(self, model_in: ModelCreate) -> LLMModel:
        """
        Create a new LLM model.
        Args:
            model_in: Model creation data
        Returns:
            Created model
        Raises:
            ProviderNotFoundException: If provider not found
            DuplicateModelException: If model name already exists for provider
        """
        # Check if provider exists
        provider = await crud_provider.get(db=self.db, id=model_in.provider_id)
        if not provider:
            raise ProviderNotFoundException(provider_id=str(model_in.provider_id))

        # Check if model name already exists for this provider
        existing_model = await crud_model.get_by_provider_and_name(
            db=self.db, provider_id=model_in.provider_id, name=model_in.name
        )
        if existing_model:
            raise DuplicateModelException(model_name=model_in.name)

        return await crud_model.create(db=self.db, obj_in=model_in)

    async def list_models(
        self,
        provider_id: UUID | None = None,
        is_active: bool | None = None,
        model_name: str | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> Sequence[LLMModel]:
        """
        List models with optional filtering.
        Args:
            provider_id: Filter by provider ID
            is_active: Filter by active status
            model_name: Filter by model name
            offset: Number of records to skip
            limit: Maximum number of records to return
        Returns:
            List of models
        """
        return await crud_model.get_multi(
            db=self.db,
            skip=offset,
            limit=limit,
            provider_id=provider_id,
            is_active=is_active,
            name=model_name,
        )

    async def list_all_models(self) -> dict[str, list[LLMModel]]:
        """
        List all models grouped by provider.
        Returns:
            Dictionary with provider names as keys and model lists as values
        """
        return await crud_model.list_models_by_provider(db=self.db)

    async def get_model(self, llm_model_id: UUID) -> LLMModel:
        """
        Get a specific model by ID.
        Args:
            llm_model_id: UUID of the model
        Returns:
            Model instance
        Raises:
            ModelNotFoundException: If model not found
        """
        model = await crud_model.get(db=self.db, id=llm_model_id)
        if not model:
            raise ModelNotFoundException(model_id=str(llm_model_id))
        return model

    async def update_model(self, llm_model_id: UUID, model_in: ModelUpdate) -> LLMModel:
        """
        Update a model.
        Args:
            llm_model_id: UUID of the model to update
            model_in: Model update data
        Returns:
            Updated model
        Raises:
            ModelNotFoundException: If model not found
            DuplicateModelException: If updated name already exists for provider
        """
        model = await crud_model.get(db=self.db, id=llm_model_id)
        if not model:
            raise ModelNotFoundException(model_id=str(llm_model_id))

        # Check for duplicate name if name is being updated
        if model_in.name and model_in.name != model.name:
            existing_model = await crud_model.get_by_provider_and_name(
                db=self.db, provider_id=model.provider_id, name=model_in.name
            )
            if existing_model:
                raise DuplicateModelException(model_name=model_in.name)

        updated_model = await crud_model.update(db=self.db, id=llm_model_id, obj_in=model_in)
        if not updated_model:
            raise ModelNotFoundException(model_id=str(llm_model_id))
        return updated_model

    async def delete_model(self, llm_model_id: UUID) -> None:
        """
        Delete a model.
        Args:
            llm_model_id: UUID of the model to delete
        Raises:
            ModelNotFoundException: If model not found
        """
        model = await crud_model.get(db=self.db, id=llm_model_id)
        if not model:
            raise ModelNotFoundException(model_id=str(llm_model_id))

        await crud_model.delete(db=self.db, id=llm_model_id)
