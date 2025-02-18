from enum import Enum
from typing import Any

from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel

from app.chat.schemas.message import MessageRead


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


class StreamBlockType(str, Enum):
    """
    Types of blocks in the message stream
    """

    CONTENT = "content"
    ERROR = "error"

    TOOL_START = "tool_start"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    THINKING = "thinking"
    DONE = "done"


class StreamBlock(BaseModel):
    """
    Base model for different types of stream blocks.
    - For thinking/content blocks: use 'content' field
    - For tool blocks: use tool_* fields
    - For errors: use error_* fields
    """

    type: StreamBlockType

    # Used for thinking and regular content blocks
    content: str | None = None

    # Tool-specific fields
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None
    tool_result: list[TextContent | ImageContent | EmbeddedResource] | None = None

    # Error handling
    error_type: str | None = None
    error_detail: str | None = None

    # Final message
    message: MessageRead | None = None


class ToolExecution(BaseModel):
    """
    Represents a single tool execution during chat completion
    """

    id: str
    name: str
    arguments: str | dict
    result: str | dict | None = None
    error: str | None = None


class CompletionMetadata(BaseModel):
    """
    Final metadata for a chat completion
    """

    tool_executions: list[ToolExecution] = []
    content: str = ""
