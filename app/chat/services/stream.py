from typing import Any

from mcp.types import EmbeddedResource, ImageContent, TextContent

from app.chat.schemas.stream import StreamBlock, StreamBlockType


class StreamBlockFactory:
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
