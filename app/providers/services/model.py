from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.crud import crud_model
from app.providers.exceptions import DuplicateModelException, ModelNotFoundException
from app.providers.models import LLMModel
from app.providers.schemas import ModelCreate, ModelUpdate
from app.providers.services.provider import LLMProviderService


class LLMModelService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _check_duplicate_name(self, provider_id: UUID, model_name: str) -> None:
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
        # Verify provider exists
        provider_service = LLMProviderService(db=self.db)
        await provider_service.get_provider(provider_id=model_in.provider_id)
        # Check if model with same name already exists for this provider
        await self._check_duplicate_name(provider_id=model_in.provider_id, model_name=model_in.name)
        return await crud_model.create(db=self.db, obj_in=model_in)

    async def list_models(
        self, provider_id: UUID | None = None, is_active: bool | None = None, offset: int = 0, limit: int = 10
    ) -> Sequence[LLMModel]:
        filters = []
        if provider_id:
            filters.append(crud_model.model.provider_id == provider_id)
        if is_active is not None:
            filters.append(crud_model.model.is_active == is_active)
        models = await crud_model.filter(db=self.db, filters=filters, offset=offset, limit=limit)
        return models

    async def get_model(self, llm_model_id: UUID) -> LLMModel:
        model = await crud_model.get(db=self.db, id=llm_model_id)
        if not model:
            raise ModelNotFoundException(model_id=llm_model_id)
        return model

    async def update_model(self, llm_model_id: UUID, model_in: ModelUpdate) -> LLMModel | None:
        llm_model = await self.get_model(llm_model_id=llm_model_id)
        if model_in.name and model_in.name != llm_model.name:
            await self._check_duplicate_name(provider_id=llm_model.provider_id, model_name=model_in.name)
        return await crud_model.update(db=self.db, id=llm_model.id, obj_in=model_in)

    async def delete_model(self, llm_model_id: UUID) -> None:
        llm_model = await self.get_model(llm_model_id=llm_model_id)
        await crud_model.delete(db=self.db, id=llm_model.id)

    async def list_all_models(self) -> dict[str, list[LLMModel]]:
        return await crud_model.list_models_by_provider(db=self.db)
