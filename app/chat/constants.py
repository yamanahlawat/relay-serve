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
    MAX_TOKENS: int = Field(default=4096, gt=0)


class SystemDefaults(BaseModel):
    """
    Default system context for chat sessions
    """

    CONTEXT: str = Field(
        default=(
            "You are a helpful AI assistant. "
            "You provide accurate, informative, and engaging responses. "
            "If you're unsure about something, you'll admit it rather than making assumptions. "
            "You aim to be concise while ensuring all relevant information is included."
        )
    )


class ErrorMessages(BaseModel):
    """
    User-friendly error messages for chat service
    """

    PROVIDER_ERROR: str = Field(
        default="I apologize, but I'm having trouble connecting to my services right now. Please try again in a few moments."
    )
    RATE_LIMIT_ERROR: str = Field(
        default="I'm currently handling too many requests. Please wait a moment before trying again."
    )
    GENERAL_ERROR: str = Field(default="I apologize, but I encountered an unexpected issue. Please try again.")


# Initialize defaults
llm_defaults = LLMDefaults()
system_defaults = SystemDefaults()
error_messages = ErrorMessages()
