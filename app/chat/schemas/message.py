from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.chat.constants import MessageRole, MessageStatus
from app.chat.schemas.common import ChatUsage


class MessageUsage(BaseModel):
    """
    Schema for message usage metrics
    """

    input_tokens: int = Field(default=0, ge=0, le=100000)
    output_tokens: int = Field(default=0, ge=0, le=100000)
    input_cost: float = Field(default=0.0, ge=0.0, le=1000.0)
    output_cost: float = Field(default=0.0, ge=0.0, le=1000.0)


class MessageCreate(BaseModel):
    """
    Schema for creating a new message
    """

    content: str = Field(min_length=1)
    role: MessageRole = Field(default=MessageRole.USER)
    status: MessageStatus = Field(default=MessageStatus.PENDING)
    parent_id: UUID | None = None
    usage: MessageUsage = Field(default_factory=MessageUsage)
    extra_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    def validate_content(cls, v: str) -> str:
        if len(v.strip()) == 0:
            raise ValueError("Content cannot be empty or just whitespace")
        return v.strip()


class MessageRead(BaseModel):
    """
    Schema for reading a message
    """

    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    status: MessageStatus
    parent_id: UUID | None = None
    created_at: datetime
    usage: ChatUsage | None = None
    error_code: str | None = None
    error_message: str | None = None
    extra_data: dict[str, Any]

    class Config:
        from_attributes = True


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
