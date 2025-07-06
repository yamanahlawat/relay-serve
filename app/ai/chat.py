"""Chat service using pydantic_ai and mem0."""

import json
from typing import Any, AsyncIterator

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.settings import ModelSettings

from app.ai.memory import memory_service
from app.ai.providers.factory import ProviderFactory
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class ChatService:
    """Service for handling chat completions with pydantic_ai and mem0."""

    def __init__(self) -> None:
        """Initialize the chat service."""
        self._agents: dict[str, Agent] = {}

    def _get_agent_key(self, provider: LLMProvider, model: LLMModel) -> str:
        """Generate a unique key for caching agents."""
        return f"{provider.name}:{model.name}"

    def _get_or_create_agent(
        self,
        provider: LLMProvider,
        model: LLMModel,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Get or create a cached agent for the provider."""
        cache_key = self._get_agent_key(provider, model)

        if cache_key not in self._agents:
            self._agents[cache_key] = ProviderFactory.create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

        return self._agents[cache_key]

    async def _prepare_conversation_context(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> str:
        """
        Prepare conversation context with memory.
        """
        context_parts = []

        # Always search for relevant memories
        relevant_memories = await memory_service.search_memories(
            query=message,
            user_id=user_id,
            session_id=session_id,
            limit=3,
        )

        # Add relevant memories to context if found
        if relevant_memories:
            memory_context = "\n".join([f"Memory: {memory.get('memory', '')}" for memory in relevant_memories])
            context_parts.append(f"Relevant memories:\n{memory_context}")

        # Add the current user message
        context_parts.append(f"User: {message}")

        return "\n\n".join(context_parts)

    async def generate_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        user_id: str,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response using pydantic_ai."""
        try:
            # Get or create agent
            agent = self._get_or_create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Prepare conversation context (memory always included)
            conversation_context = await self._prepare_conversation_context(
                user_id=user_id,
                session_id=session_id,
                message=message,
            )

            # Add user message to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=message,
                role="user",
            )

            # Prepare model settings, using model defaults if not provided
            model_settings_dict = {}
            if temperature is not None:
                model_settings_dict["temperature"] = temperature
            elif model.default_temperature is not None:
                model_settings_dict["temperature"] = model.default_temperature

            if max_tokens is not None:
                model_settings_dict["max_tokens"] = max_tokens
            elif model.default_max_tokens is not None:
                model_settings_dict["max_tokens"] = model.default_max_tokens

            # Generate response using pydantic_ai
            if model_settings_dict:
                model_settings = ModelSettings(**model_settings_dict)
                response = await agent.run(conversation_context, model_settings=model_settings)
            else:
                response = await agent.run(conversation_context)

            # Add AI response to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=str(response.output),
                role="assistant",
            )

            return str(response.output)

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise

    async def stream_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        user_id: str,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response using pydantic_ai with full event handling and markdown formatting.

        This method handles:
        - Tool calls with formatted output
        - Text streaming with real-time updates
        - Error handling with markdown formatting
        - Memory integration (always included)
        """
        try:
            # Get or create agent
            agent = self._get_or_create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Prepare conversation context (memory always included)
            conversation_context = await self._prepare_conversation_context(
                user_id=user_id,
                session_id=session_id,
                message=message,
            )

            # Add user message to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=message,
                role="user",
            )

            # Prepare model settings, using model defaults if not provided
            model_settings_dict = {}
            if temperature is not None:
                model_settings_dict["temperature"] = temperature
            elif model.default_temperature is not None:
                model_settings_dict["temperature"] = model.default_temperature

            if max_tokens is not None:
                model_settings_dict["max_tokens"] = max_tokens
            elif model.default_max_tokens is not None:
                model_settings_dict["max_tokens"] = model.default_max_tokens

            # Track the full response for memory storage
            full_response = ""
            model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else None
            content_streamed = False

            # Use pydantic_ai's iter method for full event streaming
            async with agent.iter(conversation_context, model_settings=model_settings) as run:
                async for node in run:
                    if Agent.is_user_prompt_node(node):
                        # User prompt - already handled, skip
                        continue

                    elif Agent.is_model_request_node(node):
                        # Model is processing - stream the partial responses
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent):
                                    # Start of a new part - check if it's a tool call
                                    if isinstance(event.part, ToolCallPart):
                                        # This is a tool call
                                        formatted = self._format_tool_call_start(
                                            event.part.tool_name,
                                            event.part.args if isinstance(event.part.args, dict) else {},
                                        )
                                        yield formatted
                                        full_response += formatted
                                    elif hasattr(event.part, "content"):
                                        # This is a text part with initial content
                                        text_chunk = event.part.content
                                        yield text_chunk
                                        full_response += text_chunk
                                        content_streamed = True

                                elif isinstance(event, PartDeltaEvent):
                                    if isinstance(event.delta, TextPartDelta):
                                        # Stream text content as it arrives
                                        text_chunk = event.delta.content_delta
                                        yield text_chunk
                                        full_response += text_chunk
                                        content_streamed = True
                                    elif isinstance(event.delta, ToolCallPartDelta):
                                        # Tool call arguments are being built - could show progress
                                        # For now, we'll handle this in the tool call result
                                        pass

                                elif isinstance(event, FinalResultEvent):
                                    # Model has finished generating this part
                                    if event.tool_name:
                                        # This was a tool call completion
                                        pass  # We'll handle results in CallToolsNode
                                    # For text, this just means the text part is complete

                    elif Agent.is_call_tools_node(node):
                        # Handle tool execution and results
                        async with node.stream(run.ctx) as tool_stream:
                            async for event in tool_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    # Tool is being called - already showed this in model request
                                    pass

                                elif isinstance(event, FunctionToolResultEvent):
                                    # Tool call completed with result
                                    formatted = self._format_tool_call_result(
                                        getattr(event, "tool_name", "unknown"), str(event.result.content)
                                    )
                                    yield formatted
                                    full_response += formatted

                    elif Agent.is_end_node(node):
                        # Final result - the conversation is complete
                        if content_streamed:
                            # Content was already streamed, just log the final result
                            if run.result and hasattr(run.result, "data"):
                                logger.debug(f"Final result: {run.result.data}")
                        else:
                            # No content was streamed, yield the final result
                            if run.result and hasattr(run.result, "data"):
                                final_output = str(run.result.data)
                                yield final_output
                                full_response += final_output

            # Add full AI response to memory after streaming is complete
            if full_response.strip():
                await memory_service.add_memory(
                    user_id=user_id,
                    session_id=session_id,
                    message=full_response.strip(),
                    role="assistant",
                )

        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            # Yield an error message to the client in markdown format
            error_msg = f"\n\n❌ **Error:** Unable to generate response. Please try again.\n\n*Details: {str(e)}*"
            yield error_msg
            raise

    async def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, str]]:
        """
        Get conversation history for a session using mem0.

        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of conversation messages
        """
        try:
            # Get memories for this session
            memories = await memory_service.get_memories(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
            )

            # Convert memories to conversation format
            history = []
            for memory in memories:
                history.append(
                    {
                        "role": memory.get("role", "user"),
                        "content": memory.get("memory", ""),
                        "timestamp": memory.get("created_at", ""),
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}", exc_info=True)
            return []

    async def clear_session(
        self,
        user_id: str,
        session_id: str,
    ) -> bool:
        """
        Clear all memories for a specific chat session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear memories for this session
            await memory_service.delete_session_memories(
                user_id=user_id,
                session_id=session_id,
            )

            return True

        except Exception as e:
            logger.error(f"Error clearing session: {e}", exc_info=True)
            return False

    def _format_tool_call_start(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format the start of a tool call as markdown."""
        args_str = json.dumps(args, indent=2) if args else "{}"
        return f"\n\n🔧 **Calling tool: `{tool_name}`**\n```json\n{args_str}\n```\n\n"

    def _format_tool_call_result(self, tool_name: str, result: str) -> str:
        """Format tool call result as markdown."""
        return f"✅ **Tool `{tool_name}` completed:**\n> {result}\n\n"

    def _format_thinking_start(self) -> str:
        """Format the start of AI thinking process."""
        return "🤔 **Thinking...**\n\n"

    def _format_response_start(self) -> str:
        """Format the start of the final response."""
        return "💬 **Response:**\n\n"

    def _format_markdown_response(self, text: str) -> str:
        """
        Format text as markdown for streaming.

        This method ensures that text is properly formatted for markdown rendering
        on the frontend, including code blocks, lists, and other formatting.
        """
        # For basic streaming, we return the text as-is
        # The frontend should handle markdown parsing
        return text

    def _format_streaming_chunk(self, chunk: str, chunk_type: str = "text") -> str:
        """
        Format a streaming chunk with metadata for frontend parsing.

        Args:
            chunk: The text chunk to format
            chunk_type: Type of chunk (text, tool_call, tool_result, error)

        Returns:
            Formatted chunk with metadata
        """
        if chunk_type == "text":
            # For text chunks, just return the markdown content
            return chunk
        elif chunk_type == "tool_call":
            # Tool calls are already formatted as markdown
            return chunk
        elif chunk_type == "tool_result":
            # Tool results are already formatted as markdown
            return chunk
        elif chunk_type == "error":
            # Error messages are already formatted as markdown
            return chunk
        else:
            return chunk

    def _format_thinking_chunk(self, content: str) -> str:
        """Format thinking/reasoning content as markdown."""
        return f"🤔 **Thinking:** {content}\n\n"

    def _format_code_block(self, code: str, language: str = "") -> str:
        """Format code as a markdown code block."""
        return f"```{language}\n{code}\n```\n\n"

    def _format_list_item(self, item: str, ordered: bool = False, index: int = 1) -> str:
        """Format a list item."""
        if ordered:
            return f"{index}. {item}\n"
        else:
            return f"- {item}\n"

    def _format_blockquote(self, text: str) -> str:
        """Format text as a blockquote."""
        lines = text.split("\n")
        return "\n".join(f"> {line}" for line in lines) + "\n\n"

    def _format_header(self, text: str, level: int = 1) -> str:
        """Format text as a markdown header."""
        return f"{'#' * level} {text}\n\n"

    def _format_emphasis(self, text: str, strong: bool = False) -> str:
        """Format text with emphasis."""
        if strong:
            return f"**{text}**"
        else:
            return f"*{text}*"

    def _format_link(self, text: str, url: str) -> str:
        """Format a markdown link."""
        return f"[{text}]({url})"

    def _format_table_row(self, columns: list[str], is_header: bool = False) -> str:
        """Format a table row in markdown."""
        row = "| " + " | ".join(columns) + " |"
        if is_header:
            separator = "| " + " | ".join(["---"] * len(columns)) + " |"
            return f"{row}\n{separator}\n"
        else:
            return f"{row}\n"


chat_service = ChatService()
