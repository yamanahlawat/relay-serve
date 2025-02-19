from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel

from app.chat.constants import llm_defaults


class ModelBase(BaseModel):
    """
    Base schema for LLM models.
    """

    name: str
    is_active: bool = True
    max_tokens: int = Field(default=llm_defaults.MAX_TOKENS, gt=0)
    temperature: float = Field(default=llm_defaults.TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=llm_defaults.TOP_P, ge=0.0, le=1.0)
    config: dict[str, Any] = Field(default_factory=dict)
    input_cost_per_token: float = Field(default=0.0, ge=0.0)
    output_cost_per_token: float = Field(default=0.0, ge=0.0)


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
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    config: dict[str, Any] | None = None
    input_cost_per_token: float | None = Field(default=None, ge=0.0)
    output_cost_per_token: float | None = Field(default=None, ge=0.0)


class ModelsByProvider(RootModel[dict[str, list[ModelRead]]]):
    """
    Schema for reading models grouped by provider.
    Args:
        RootModel (dict[str, list[ModelRead]]): Root model for models grouped by provider
    """

    class Config:
        from_attributes = True
