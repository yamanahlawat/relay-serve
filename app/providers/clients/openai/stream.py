from typing import Any, AsyncGenerator, Tuple
from uuid import UUID

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from app.chat.schemas.stream import StreamBlock
from app.chat.services.sse import get_sse_manager
from app.chat.services.stream import StreamBlockFactory


class OpenAIStreamHandler:
    """
    Handles streaming operations for OpenAI provider.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def should_cancel(self, session_id: UUID | None) -> bool:
        """
        Check if the stream should be cancelled for a given session.
        Args:
            session_id: Optional UUID of the chat session
        Returns:
            bool: True if stream should be cancelled, False otherwise
        """
        if not session_id:
            return False

        try:
            sse_manager = await get_sse_manager()
            cancel_key = f"sse:cancel:{session_id}"
            return await sse_manager.redis.exists(cancel_key)
        except Exception as error:
            logger.error(f"Error checking cancel state: {error}")
            return False

    def handle_completion(self, chunk: ChatCompletionChunk) -> Tuple[Tuple[int, int] | None, bool]:
        """
        Handle completion/finish of a stream chunk and manage usage.
        Args:
            chunk: Response chunk from OpenAI API
        Returns:
            Tuple of (UsageInfo | None, should_stop: bool)
        """
        # Directly checking finish_reason first
        if chunk.choices and chunk.choices[0].finish_reason == "stop":
            if hasattr(chunk, "usage") and chunk.usage:
                return (chunk.usage.prompt_tokens, chunk.usage.completion_tokens), True
            return None, False

        # Then check for usage-only chunks
        if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
            return (chunk.usage.prompt_tokens, chunk.usage.completion_tokens), True

        return None, False

    def handle_tool_calls(
        self,
        delta: Any,
        current_tool_calls: dict[str, dict],
        current_tool_index: dict[int, str],
    ) -> Tuple[dict[str, dict], dict[int, str], StreamBlock | None]:
        """
        Process tool calls from a chunk delta.
        Args:
            delta: Chunk delta from OpenAI API
            current_tool_calls: Current state of tool calls
            current_tool_index: Mapping of tool indices to IDs
        Returns:
            Tuple of (updated_tool_calls, updated_tool_index, stream_block)
        """
        if not (tool_calls := getattr(delta, "tool_calls", [])):
            return current_tool_calls, current_tool_index, None

        for tool_call in tool_calls:
            # Get existing tool_id from index if this is a continuation
            tool_id = tool_call.id or current_tool_index.get(tool_call.index)

            if tool_id and tool_id not in current_tool_calls:
                # This is a new tool call
                current_tool_calls[tool_id] = {
                    "id": tool_id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "",
                    "type": tool_call.type or "function",
                }
                current_tool_index[tool_call.index] = tool_id

                # Only emit thinking block if we have the tool name
                if tool_call.function.name:
                    return (
                        current_tool_calls,
                        current_tool_index,
                        StreamBlockFactory.create_thinking_block(
                            content=f"Preparing to use tool: {tool_call.function.name}"
                        ),
                    )
            elif tool_id and tool_call.function.arguments:
                # Accumulate arguments for existing tool call
                current_tool_calls[tool_id]["arguments"] += tool_call.function.arguments

        return current_tool_calls, current_tool_index, None

    async def create_completion_stream(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Create a completion stream from OpenAI.
        Args:
            model: Model identifier
            messages: Formatted messages for the API
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            top_p: Top-p parameter
            tools: Optional tool definitions
        Returns:
            AsyncGenerator yielding completion chunks
        """
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            tools=tools,
            tool_choice="auto" if tools else None,
            stream_options={"include_usage": True},
        )
