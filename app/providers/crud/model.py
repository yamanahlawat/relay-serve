from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.database.crud import CRUDBase
from app.providers.models import LLMModel, LLMProvider
from app.providers.schemas import ModelCreate, ModelUpdate


class CRUDModel(CRUDBase[LLMModel, ModelCreate, ModelUpdate]):
    """
    CRUD operations for LLM Models.
    """

    async def list_models_by_provider(self, db: AsyncSession) -> dict[str, list[LLMModel]]:
        """
        List all models grouped by provider name.

        Returns:
            dict[str, Sequence[LLMModel]]: Dictionary with provider names as keys and their models as values
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
