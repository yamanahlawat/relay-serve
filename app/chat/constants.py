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


class ErrorCode(BaseEnum):
    """
    Error codes for chat operations
    """

    # Session not found or invalid state
    INVALID_SESSION = "invalid_session"

    # Message not found or invalid format
    INVALID_MESSAGE = "invalid_message"

    # Too many requests in time period
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Input + context exceeds model's max tokens
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"

    # LLM provider returned an error
    PROVIDER_ERROR = "provider_error"
