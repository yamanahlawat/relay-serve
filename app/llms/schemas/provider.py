from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, SecretStr

from app.llms.constants import ProviderType


class ProviderBase(BaseModel):
    """
    Base schema for LLM providers.
    """

    name: str
    type: ProviderType
    is_active: bool = True
    base_url: str | None = None


class ProviderCreate(ProviderBase):
    """
    Schema for creating a new provider.
    """

    api_key: SecretStr | None = None

    model_config = ConfigDict(json_encoders={SecretStr: lambda v: v.get_secret_value() if v else None})


class ProviderRead(ProviderBase):
    """
    Schema for reading a provider.
    """

    id: UUID
    api_key: SecretStr | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProviderUpdate(BaseModel):
    """
    Schema for updating a provider.
    """

    name: str | None = None
    type: ProviderType | None = None
    is_active: bool | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None

    model_config = ConfigDict(json_encoders={SecretStr: lambda v: v.get_secret_value() if v else None})
