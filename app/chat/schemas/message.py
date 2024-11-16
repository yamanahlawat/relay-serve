from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.chat.constants import MessageRole, MessageStatus
from app.chat.schemas.common import ChatUsage


class MessageCreate(BaseModel):
    """
    Schema for creating a new message
    """

    content: str = Field(min_length=1)
    role: MessageRole = Field(default=MessageRole.USER)
    status: MessageStatus = Field(default=MessageStatus.PENDING)
    parent_id: UUID | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    input_cost: float = Field(default=0.0, ge=0.0)
    output_cost: float = Field(default=0.0, ge=0.0)
    extra_data: dict[str, Any] = Field(default_factory=dict)


class MessageRead(BaseModel):
    """
    Schema for reading a message
    """

    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    status: MessageStatus
    parent_id: UUID | None
    created_at: datetime
    usage: ChatUsage
    error_code: str | None = None
    error_message: str | None = None
    extra_data: dict[str, Any]

    class Config:
        from_attributes = True


class MessageUpdate(BaseModel):
    """
    Schema for updating a message
    """

    content: str | None = None
    status: MessageStatus | None = None
    extra_data: dict[str, Any] | None = None
