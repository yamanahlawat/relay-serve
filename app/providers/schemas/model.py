from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBase(BaseModel):
    """
    Base schema for LLM models.
    """

    name: str
    is_active: bool = True
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
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

    is_active: bool | None = None
    max_tokens: int | None = Field(None, gt=0)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    config: dict[str, Any] | None = None
    input_cost_per_token: float | None = Field(None, ge=0.0)
    output_cost_per_token: float | None = Field(None, ge=0.0)
