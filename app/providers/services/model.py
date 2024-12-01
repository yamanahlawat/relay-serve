from typing import Sequence
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.providers.crud import crud_model, crud_provider
from app.providers.models.model import LLMModel
from app.providers.schemas import ModelCreate
from app.providers.schemas.model import ModelUpdate


class LLMModelService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_model(self, model_in: ModelCreate) -> LLMModel:
        # Verify provider exists
        provider = await crud_provider.get(db=self.db, id=model_in.provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider with id {model_in.provider_id} not found",
            )

        # Check if model with same name exists for this provider
        filters = [crud_model.model.provider_id == model_in.provider_id, crud_model.model.name == model_in.name]
        existing_model = await crud_model.filter(
            db=self.db,
            filters=filters,
        )
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model {model_in.name} already exists for this provider",
            )

        return await crud_model.create(db=self.db, obj_in=model_in)

    async def list_models(
        self, provider_id: UUID | None = None, offset: int = 0, limit: int = 100
    ) -> Sequence[LLMModel]:
        if provider_id:
            filters = [crud_model.model.provider_id == provider_id]
            models = await crud_model.filter(
                db=self.db,
                filters=filters,
                offset=offset,
                limit=limit,
            )
        else:
            models = await crud_model.filter(db=self.db, offset=offset, limit=limit)
        return models

    async def get_model(self, llm_model_id: UUID) -> LLMModel:
        model = await crud_model.get(db=self.db, id=llm_model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with id {llm_model_id} not found",
            )
        return model

    async def update_model(self, llm_model_id: UUID, model_in: ModelUpdate) -> LLMModel | None:
        llm_model = await self.get_model(llm_model_id=llm_model_id)
        if model_in.name and model_in.name != llm_model.name:
            # Check if model with new name already exists for this provider
            filters = [
                crud_model.model.provider_id == llm_model.provider_id,
                crud_model.model.name == model_in.name,
            ]
            existing_model = await crud_model.filter(db=self.db, filters=filters)
            if existing_model:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Model {model_in.name} already exists for this provider",
                )
        return await crud_model.update(db=self.db, id=llm_model.id, obj_in=model_in)

    async def delete_model(self, llm_model_id: UUID) -> None:
        llm_model = await self.get_model(llm_model_id=llm_model_id)
        await crud_model.delete(db=self.db, id=llm_model.id)
