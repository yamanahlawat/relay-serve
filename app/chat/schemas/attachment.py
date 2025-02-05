from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.chat.constants import AttachmentType
from app.storages.utils import get_attachment_download_url


class AttachmentBase(BaseModel):
    """
    Base schema with shared attributes for Attachment
    """

    file_name: Annotated[str, Field(max_length=255)]
    file_size: int = Field(gt=0)
    mime_type: Annotated[str, Field(max_length=100)]
    type: AttachmentType
    storage_path: Annotated[str, Field(max_length=500)]


class AttachmentCreate(AttachmentBase):
    """
    Schema for creating a new Attachment.
    Includes the required message_id.
    """

    message_id: UUID


class AttachmentUpdate(BaseModel):
    """
    Schema for updating an Attachment.
    All fields are optional.
    """

    file_name: Annotated[str, Field(max_length=255)] | None = None
    file_size: int | None = None
    mime_type: Annotated[str, Field(max_length=100)] | None = None
    type: AttachmentType | None = None
    storage_path: Annotated[str, Field(max_length=500)] | None = None

    model_config = ConfigDict(from_attributes=True)


class AttachmentRead(AttachmentBase):
    """
    Schema for reading an Attachment.
    Includes all base fields plus the id and timestamps.
    """

    id: UUID
    message_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def absolute_url(self) -> str:
        """
        Constructs the absolute URL for this attachment.
        If using local storage, it builds the URL from settings.BASE_URL, settings.API_URL, and the stored path.
        """
        return get_attachment_download_url(storage_path=self.storage_path)
