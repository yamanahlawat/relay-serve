from pydantic import BaseModel, Field

from app.core.constants import BaseEnum


class MessageRole(BaseEnum):
    """
    Message role types
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(BaseEnum):
    """
    Message processing status
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionStatus(BaseEnum):
    """
    Chat session status
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class LLMDefaults(BaseModel):
    """
    Default parameters for LLM requests
    """

    TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=1024, gt=0)


# Initialize LLM defaults
llm_defaults = LLMDefaults()
