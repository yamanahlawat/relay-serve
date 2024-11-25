from uuid import UUID

from pydantic import BaseModel, Field

from app.chat.schemas.common import ChatUsage
from app.core.config import settings


class CompletionParams(BaseModel):
    """
    Parameters for completion generation
    """

    max_tokens: int = Field(default=settings.DEFAULT_MAX_TOKENS, gt=0)
    temperature: float = Field(default=settings.DEFAULT_TEMPERATURE, ge=0.0, le=2.0)


class CompletionRequest(BaseModel):
    """
    Schema for chat request
    """

    provider_id: UUID
    llm_model_id: UUID
    prompt: str
    parent_id: UUID | None = None
    max_tokens: int = Field(default=settings.DEFAULT_MAX_TOKENS, gt=0)
    temperature: float = Field(default=settings.DEFAULT_TEMPERATURE, ge=0.0, le=2.0)


class CompletionResponse(BaseModel):
    """
    Schema for chat response
    """

    content: str
    model: str
    provider: str
    usage: ChatUsage
