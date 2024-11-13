from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.providers.constants import ProviderType


class ProviderBase(BaseModel):
    """
    Base schema for LLM providers.
    """

    name: ProviderType
    is_active: bool = True
    base_url: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ProviderCreate(ProviderBase):
    """
    Schema for creating a new provider.
    """

    api_key: SecretStr = Field(min_length=1)

    model_config = ConfigDict(json_encoders={SecretStr: lambda v: v.get_secret_value() if v else None})


class ProviderRead(ProviderBase):
    """
    Schema for reading a provider.
    """

    id: UUID
    api_key: SecretStr
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProviderUpdate(BaseModel):
    """
    Schema for updating a provider.
    """

    is_active: bool | None = None
    api_key: SecretStr | None = Field(None, min_length=1)
    base_url: str | None = None
    config: dict[str, Any] | None = None
