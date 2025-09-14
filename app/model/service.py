from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.model.crud import crud_model
from app.model.exceptions import DuplicateModelException, ModelNotFoundException
from app.model.model import LLMModel
from app.model.schema import ModelCreate, ModelUpdate
from app.provider.service import LLMProviderService


class LLMModelService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _check_duplicate_name(self, provider_id: UUID, model_name: str) -> None:
        """
        Check for duplicate model names for a given provider.
        Args:
            provider_id (UUID): The ID of the provider.
            model_name (str): The name of the model.
        Raises:
            DuplicateModelException: If a model with the same name already exists for the provider.
        """
        # Check if model with same name exists for this provider
        filters = [
            crud_model.model.provider_id == provider_id,
            crud_model.model.name == model_name,
        ]
        existing_model = await crud_model.filter(
            db=self.db,
            filters=filters,
        )
        if existing_model:
            raise DuplicateModelException(name=model_name)

    async def create_model(self, model_in: ModelCreate) -> LLMModel:
        """
        Create a new LLM model.
        Args:
            model_in (ModelCreate): The model creation data.
        Returns:
            LLMModel: The created model.
        """
        # Verify provider exists
        provider_service = LLMProviderService(db=self.db)
        await provider_service.get_provider(provider_id=model_in.provider_id)
        # Check if model with same name already exists for this provider
        await self._check_duplicate_name(provider_id=model_in.provider_id, model_name=model_in.name)
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
            provider_id (UUID | None, optional): Filter by provider ID. Defaults to None.
            is_active (bool | None, optional): Filter by active status. Defaults to None.
            model_name (str | None, optional): Filter by model name. Defaults to None.
            offset (int, optional): Number of records to skip. Defaults to 0.
            limit (int, optional): Maximum number of records to return. Defaults to 10.
        Returns:
            Sequence[LLMModel]: List of models.
        """
        filters = []
        if provider_id:
            filters.append(crud_model.model.provider_id == provider_id)
        if is_active is not None:
            filters.append(crud_model.model.is_active == is_active)
        if model_name:
            filters.append(crud_model.model.name.ilike(f"%{model_name}%"))
        models = await crud_model.filter(db=self.db, filters=filters, offset=offset, limit=limit)
        return models

    async def get_model(self, llm_model_id: UUID) -> LLMModel:
        """
        Get a specific LLM model by its ID.
        Args:
            llm_model_id (UUID): The ID of the model.
        Raises:
            ModelNotFoundException: If the model is not found.
        Returns:
            LLMModel: The requested model.
        """
        model = await crud_model.get(db=self.db, id=llm_model_id)
        if not model:
            raise ModelNotFoundException(model_id=llm_model_id)
        return model

    async def update_model(self, llm_model_id: UUID, model_in: ModelUpdate) -> LLMModel | None:
        """
        Update an existing LLM model.
        Args:
            llm_model_id (UUID): The ID of the model to update.
            model_in (ModelUpdate): The updated model data.
        Returns:
            LLMModel | None: The updated model or None if not found.
        """
        llm_model = await self.get_model(llm_model_id=llm_model_id)
        if model_in.name and model_in.name != llm_model.name:
            await self._check_duplicate_name(provider_id=llm_model.provider_id, model_name=model_in.name)
        return await crud_model.update(db=self.db, id=llm_model.id, obj_in=model_in)

    async def delete_model(self, llm_model_id: UUID) -> None:
        """
        Delete an LLM model by its ID.
        Args:
            llm_model_id (UUID): The ID of the model to delete.
        """
        llm_model = await self.get_model(llm_model_id=llm_model_id)
        await crud_model.delete(db=self.db, id=llm_model.id)

    async def list_all_models(self) -> dict[str, list[LLMModel]]:
        """
        List all LLM models grouped by provider.
        Returns:
            dict[str, list[LLMModel]]: A dictionary where the keys are provider IDs and the values are lists of models.
        """
        return await crud_model.list_models_by_provider(db=self.db)
