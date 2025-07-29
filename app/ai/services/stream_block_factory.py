from typing import Any

from mcp.types import EmbeddedResource, ImageContent, TextContent

from app.ai.schemas.stream import StreamBlock, StreamBlockType


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
        Create a thinking block to indicate processing state (UI status, not stored)
        """
        return StreamBlock(
            type=StreamBlockType.THINKING,
            content=content or "Thinking...",
        )

    @staticmethod
    def create_reasoning_block(
        content: str,
    ) -> StreamBlock:
        """
        Create a reasoning block from model's actual reasoning (ThinkingPart).
        This content is stored and displayed as actual model reasoning.

        Args:
            reasoning_content: The model's reasoning text
            is_delta: Whether this is a partial update (streaming)
        """
        return StreamBlock(
            type=StreamBlockType.REASONING,
            content=content,
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
        )

    @staticmethod
    def create_tool_result_block(
        tool_result: list[TextContent | ImageContent | EmbeddedResource],
        tool_call_id: str,
        tool_name: str,
    ) -> StreamBlock:
        """
        Create a block containing tool execution results
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_RESULT, tool_name=tool_name, tool_call_id=tool_call_id, tool_result=tool_result
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
    def create_done_block(content: str | None = None) -> StreamBlock:
        """
        Create a block indicating stream completion

        Args:
            content: Optional final output content from the LLM
        """
        return StreamBlock(type=StreamBlockType.DONE, content=content)

    @staticmethod
    def create_part_start_block(part_index: int, part_type: str, part_info: str) -> StreamBlock:
        """
        Create a block for part start event
        """
        return StreamBlock(
            type=StreamBlockType.THINKING,
            content=f"Starting part {part_index} ({part_type}): {part_info}",
        )

    @staticmethod
    def create_text_delta_block(content_delta: str) -> StreamBlock:
        """
        Create a block for text delta content
        """
        return StreamBlock(
            type=StreamBlockType.CONTENT,
            content=content_delta,
        )

    @staticmethod
    def create_tool_args_delta_block(
        tool_name: str,
        tool_call_id: str,
        args_delta: str,
    ) -> StreamBlock:
        """
        Create a block for tool arguments delta chunks.

        Streams raw delta chunks in the args_delta field so frontend knows exactly
        where to look for streaming arguments vs final accumulated arguments.

        Args:
            tool_name: Name of the tool
            tool_call_id: Unique identifier for the tool call
            args_delta: The delta string chunk being streamed
            current_args: Not used - pydantic-ai handles accumulation

        Returns:
            StreamBlock with delta chunk in args_delta field
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_CALL,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_delta=args_delta,  # Stream the raw delta chunk here
        )

    @staticmethod
    def create_function_tool_call_event_block(
        tool_name: str,
        tool_call_id: str,
        tool_args: dict[str, Any],
    ) -> StreamBlock:
        """
        Create a block for function tool call event with final accumulated arguments.

        This is sent when pydantic-ai has completed accumulating all the argument deltas
        and is ready to execute the tool with the final arguments.

        Args:
            tool_name: Name of the tool
            tool_call_id: Unique identifier for the tool call
            tool_args: Final accumulated arguments as a dictionary

        Returns:
            StreamBlock with complete tool_args for the frontend to use
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_CALL,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
            content=f"Calling tool {tool_name} with args: {tool_args}",
        )

    @staticmethod
    def create_function_tool_result_event_block(
        tool_call_id: str,
        tool_name: str,
        result_content: str,
    ) -> StreamBlock:
        """
        Create a block for function tool result event
        """
        return StreamBlock(
            type=StreamBlockType.TOOL_RESULT,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_result=[TextContent(type="text", text=result_content)],
            content=f"Tool {tool_name} result: {result_content}",
        )

    @staticmethod
    def create_final_result_event_block(tool_name: str | None = None) -> StreamBlock:
        """
        Create a block for final result event
        """
        content = "Finalizing response..."
        if tool_name:
            content = f"Completing {tool_name} operation..."

        return StreamBlock(
            type=StreamBlockType.THINKING,
            content=content,
        )

    @staticmethod
    def create_call_tools_node_start_block() -> StreamBlock:
        """
        Create a block for call tools node start
        """
        return StreamBlock(
            type=StreamBlockType.THINKING,
            content="Executing tools and gathering information...",
        )
