from typing import Any

from pydantic import BaseModel, Field

from app.chat.constants import AttachmentType


class AttachmentBase(BaseModel):
    """
    Base schema for attachment data
    """

    file_name: str
    file_size: int
    mime_type: str
    type: AttachmentType
    metadata: dict[str, Any] = Field(default_factory=dict)
