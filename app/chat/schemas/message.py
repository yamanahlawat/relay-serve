import json
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field, field_validator

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


class MessageInBase(BaseModel):
    """
    Base schema for creating a new message
    """

    content: str | None = None
    role: MessageRole = MessageRole.USER
    status: MessageStatus = MessageStatus.PENDING
    parent_id: UUID | None = None
    usage: MessageUsage = Field(default_factory=MessageUsage)
    extra_data: dict[str, Any] = Field(default_factory=dict)


class MessageIn(MessageInBase):
    """
    Schema for creating a new message from API
    """

    attachments: list[UploadFile] = Form(default_factory=list)

    @classmethod
    def as_form(
        cls,
        content: Annotated[str | None, Form()] = None,
        role: Annotated[MessageRole, Form()] = MessageRole.USER,
        status: Annotated[MessageStatus, Form()] = MessageStatus.PENDING,
        parent_id: Annotated[str | None, Form()] = None,
        usage: Annotated[str, Form()] = "{}",
        attachments: Annotated[list[UploadFile], File()] = [],
        extra_data: Annotated[str, Form()] = "{}",
    ) -> "MessageIn":
        """
        Convert form data to MessageIn model
        """
        try:
            # Parse parent_id to UUID if provided
            parent_uuid = UUID(parent_id) if parent_id else None

            # Parse JSON strings
            usage_dict = json.loads(usage)
            extra_data_dict = json.loads(extra_data)

            return cls(
                content=content,
                role=role,
                status=status,
                parent_id=parent_uuid,
                usage=MessageUsage(**usage_dict),
                attachments=attachments,
                extra_data=extra_data_dict,
            )
        except ValueError as e:
            raise ValueError(f"Invalid UUID format: {e}")


class MessageCreate(MessageInBase):
    """
    Schema for creating a new message
    """

    attachments: list[dict[str, Any]] = Field(default_factory=list)


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
    attachments: list[AttachmentRead] = Field(default_factory=list)
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
