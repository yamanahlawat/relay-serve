from enum import Enum

from pydantic import BaseModel


class StreamEvent(str, Enum):
    """
    Stream event types
    """

    MESSAGE = "message"  # Content chunk
    DONE = "done"  # Stream complete
    ERROR = "error"  # Error occurred


class StreamResponse(BaseModel):
    """
    Structured stream response
    """

    event: StreamEvent
    data: str | None = None
    error: str | None = None
