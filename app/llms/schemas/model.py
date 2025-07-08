"""Model schemas for validation and serialization."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel


class ModelBase(BaseModel):
    """Base schema for LLM models."""

    name: str
    is_active: bool = True
    context_length: int | None = None
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int | None = Field(default=None, gt=0)
    default_top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class ModelCreate(ModelBase):
    """Schema for creating a new model."""

    provider_id: UUID


class ModelRead(ModelBase):
    """Schema for reading a model."""

    id: UUID
    provider_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelUpdate(BaseModel):
    """Schema for updating a model."""

    name: str | None = None
    is_active: bool | None = None
    context_length: int | None = None
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    default_max_tokens: int | None = Field(default=None, gt=0)
    default_top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelsByProvider(RootModel[dict[str, list[ModelRead]]]):
    """Schema for reading models grouped by provider."""

    model_config = ConfigDict(from_attributes=True)
