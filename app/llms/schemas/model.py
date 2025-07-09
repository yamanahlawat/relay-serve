from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel

from app.chat.constants import llm_defaults


class ModelBase(BaseModel):
    """
    Base schema for LLM models.
    """

    name: str
    is_active: bool = True
    default_max_tokens: int = Field(default=llm_defaults.MAX_TOKENS, gt=0)
    default_temperature: float = Field(default=llm_defaults.TEMPERATURE, ge=0.0, le=2.0)
    default_top_p: float = Field(default=llm_defaults.TOP_P, ge=0.0, le=1.0)


class ModelCreate(ModelBase):
    """
    Schema for creating a new model.
    """

    provider_id: UUID


class ModelRead(ModelBase):
    """
    Schema for reading a model
    """

    id: UUID
    provider_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelUpdate(BaseModel):
    """
    Schema for updating a model.
    """

    name: str | None = None
    is_active: bool | None = None
    default_max_tokens: int | None = Field(default=None, gt=0)
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    default_top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelsByProvider(RootModel[dict[str, list[ModelRead]]]):
    """
    Schema for reading models grouped by provider.
    Args:
        RootModel (dict[str, list[ModelRead]]): Root model for models grouped by provider
    """

    model_config = ConfigDict(from_attributes=True)
