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
