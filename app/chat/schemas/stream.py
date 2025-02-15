from enum import Enum
from typing import Any

from mcp.types import EmbeddedResource, ImageContent, TextContent
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
    Base model for stream blocks
    """

    type: StreamBlockType
    content: str | list[TextContent | ImageContent | EmbeddedResource] | None = None

    # Tool specific fields
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None
    tool_status: str | None = None

    # Error details
    error_type: str | None = None
    error_detail: str | None = None


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
