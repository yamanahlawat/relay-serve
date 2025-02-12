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


class StreamManager:
    """
    Manages creation and handling of stream blocks.
    Provides a consistent interface for all providers.
    """

    @staticmethod
    def create_content_block(content: str) -> StreamBlock:
        """
        Create a content block for regular message content
        """
        return StreamBlock(type=StreamBlockType.CONTENT, content=content)

    @staticmethod
    def create_thinking_block(
        content: str | None = None,
    ) -> StreamBlock:
        """
        Create a thinking block to indicate processing state
        """
        return StreamBlock(
            type=StreamBlockType.THINKING,
            content=content or "Thinking...",
        )

    @staticmethod
    def create_tool_start_block(
        tool_name: str,
        tool_call_id: str,
    ) -> StreamBlock:
        """
        Create a block indicating tool execution is starting
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_START,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            content=f"Starting tool: {tool_name}",
        )

    @staticmethod
    def create_tool_call_block(
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
    ) -> StreamBlock:
        """
        Create a block for tool call with arguments
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_CALL,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=f"Calling tool: {tool_name}",
        )

    @staticmethod
    def create_tool_result_block(
        content: str | list[TextContent | ImageContent | EmbeddedResource],
        tool_call_id: str,
        tool_name: str | None = None,
    ) -> StreamBlock:
        """
        Create a block containing tool execution results
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_RESULT,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    @staticmethod
    def create_error_block(
        error_type: str,
        error_detail: str,
    ) -> StreamBlock:
        """
        Create a block for error conditions
        """
        return StreamBlock(
            type=StreamBlockType.ERROR,
            error_type=error_type,
            error_detail=error_detail,
            content=f"Error: {error_detail}",
        )

    @staticmethod
    def create_done_block() -> StreamBlock:
        """
        Create a block indicating stream completion
        """
        return StreamBlock(type=StreamBlockType.DONE, content="Stream completed")
