from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.chat.constants import MessageRole, MessageStatus
from app.chat.schemas import AttachmentRead
from app.chat.schemas.common import ChatUsage


class MessageUsage(BaseModel):
    """
    Schema for message usage metrics
    """

    input_tokens: int = Field(default=0, ge=0, le=100000)
    output_tokens: int = Field(default=0, ge=0, le=100000)
    input_cost: float = Field(default=0.0, ge=0.0, le=1000.0)
    output_cost: float = Field(default=0.0, ge=0.0, le=1000.0)


class MessageBase(BaseModel):
    """
    Base schema for creating a new message
    """

    content: str | None = None
    role: MessageRole = MessageRole.USER
    status: MessageStatus = MessageStatus.PENDING
    parent_id: UUID | None = None
    usage: MessageUsage = Field(default_factory=MessageUsage)
    extra_data: dict[str, Any] = Field(default_factory=dict)


class MessageCreate(MessageBase):
    """
    Schema for creating a new message
    """

    attachment_ids: list[UUID] = Field(default_factory=list, description="List of attachment IDs already uploaded")

    @model_validator(mode="after")
    def check_content_or_attachments(self) -> Self:
        if not self.content and not self.attachment_ids:
            raise ValueError("Either non-empty content or at least one attachment must be provided.")
        return self


class MessageRead(BaseModel):
    """
    Schema for reading a message
    """

    id: UUID
    session_id: UUID
    role: MessageRole
    content: str | None = None
    status: MessageStatus
    parent_id: UUID | None = None
    created_at: datetime
    usage: ChatUsage | None = None
    attachments: list[AttachmentRead] = Field(
        default_factory=list,
        serialization_alias="attachments",
        alias="direct_attachments",
    )
    error_code: str | None = None
    error_message: str | None = None
    extra_data: dict[str, Any]

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class MessageUpdate(BaseModel):
    """
    Schema for updating a message
    """

    content: str | None = Field(default=None, min_length=1)
    status: MessageStatus | None = None
    extra_data: dict[str, Any] | None = None

    @field_validator("content")
    def validate_content(cls, v: str | None) -> str | None:
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Content cannot be empty or just whitespace")
        return v.strip() if v is not None else None
