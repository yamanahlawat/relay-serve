from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.session.constants import SessionStatus


class ChatUsage(BaseModel):
    """
    Schema for token usage and costs
    """

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float = Field(description="Total cost (input + output)")


class SessionCreate(BaseModel):
    """
    Schema for creating a new chat session
    """

    title: str = Field(min_length=1, max_length=255)
    system_context: str | None = Field(default=None)
    provider_id: UUID
    llm_model_id: UUID
    extra_data: dict[str, Any] = Field(default_factory=dict)


class SessionRead(BaseModel):
    """
    Schema for reading a chat session
    """

    id: UUID
    title: str
    status: SessionStatus
    system_context: str | None
    provider_id: UUID
    llm_model_id: UUID
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None
    usage: ChatUsage | None = None
    extra_data: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class SessionUpdate(BaseModel):
    """
    Schema for updating a chat session
    """

    title: str | None = None
    status: SessionStatus | None = None
    system_context: str | None = None
    provider_id: UUID | None = None
    llm_model_id: UUID | None = None
    extra_data: dict[str, Any] | None = None
