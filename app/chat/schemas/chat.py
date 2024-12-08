from uuid import UUID

from pydantic import BaseModel, Field

from app.chat.constants import llm_defaults
from app.chat.schemas.common import ChatUsage


class CompletionParams(BaseModel):
    """
    Parameters for completion generation
    """

    max_tokens: int = Field(default=llm_defaults.MAX_TOKENS, gt=0)
    temperature: float = Field(default=llm_defaults.TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=llm_defaults.TOP_P, ge=0.0, le=1.0)


class CompletionRequest(BaseModel):
    """
    Schema for chat request
    """

    provider_id: UUID
    llm_model_id: UUID
    prompt: str
    parent_id: UUID | None = None
    model_params: CompletionParams = CompletionParams()


class CompletionResponse(BaseModel):
    """
    Schema for chat response
    """

    content: str
    model: str
    provider: str
    usage: ChatUsage
