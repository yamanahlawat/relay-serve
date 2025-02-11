import json
from typing import Any, AsyncGenerator, Sequence
from uuid import UUID

from loguru import logger
from mcp.types import EmbeddedResource, ImageContent, TextContent
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.chat.constants import AttachmentType, MessageRole
from app.chat.models import ChatMessage
from app.chat.schemas.stream import StreamBlock, StreamManager
from app.chat.services.sse import get_sse_manager
from app.files.image.processor import ImageProcessor
from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult
from app.model_context_protocol.services.tool import MCPService
from app.providers.clients.base import LLMProviderBase
from app.providers.clients.openai.schemas import OpenAIFunction, OpenAIFunctionParameters, OpenAITool
from app.providers.constants import ProviderType
from app.providers.exceptions import (
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from app.providers.factory import ProviderFactory
from app.providers.models import LLMProvider


class OpenAIProvider(LLMProviderBase):
    """
    OpenAI provider implementation using the official Python SDK.
    """

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self._client = AsyncOpenAI(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url or None,
        )
        self._last_usage = None

    def _handle_api_error(self, error: APIError) -> None:
        """
        Handle OpenAI API errors and raise appropriate provider exceptions.
        """
        if isinstance(error, APIConnectionError):
            logger.exception("OpenAI API connection error during generation")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error=str(error),
            ) from error
        elif isinstance(error, RateLimitError):
            logger.exception("OpenAI API rate limit exceeded during generation")
            raise ProviderRateLimitError(
                provider=self.provider_type,
                error=str(error),
            ) from error
        elif isinstance(error, AuthenticationError):
            logger.exception("OpenAI authentication error during generation")
            raise ProviderConfigurationError(
                provider=self.provider_type,
                message="Invalid API credentials",
                error=str(error),
            ) from error
        elif isinstance(error, APIStatusError):
            logger.exception("OpenAI API status error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=error.status_code,
                message=error.message,
                error=str(error),
            ) from error
        elif isinstance(error, APIError):
            logger.exception("OpenAI API error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=getattr(error, "status_code", 500),
                error=str(error),
            ) from error

    def _format_message_content(self, message: ChatMessage, is_current: bool = False) -> list[dict]:
        """
        Format message content with any image attachments.
        """
        content = []

        # Add text content
        content.append({"type": "text", "text": message.content})

        # Add any image attachments
        if is_current and message.attachments:
            for attachment in message.direct_attachments:
                if attachment.type == AttachmentType.IMAGE.value:
                    base64_image = ImageProcessor.encode_image_to_base64(image_path=attachment.storage_path)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        }
                    )

        return content

    def _prepare_messages(
        self,
        messages: Sequence[ChatMessage],
        current_message: ChatMessage,
        system_context: str,
    ) -> list[dict[str, str]]:
        """
        Prepare message history for OpenAI API.
        Args:
            messages: Previous messages in the conversation
            current_message: Current message to generate completion for
            system_context: System context/instructions
        Returns:
            List of formatted messages for the OpenAI API
        """

        formatted_messages = []

        if system_context:
            formatted_messages.append({"role": "system", "content": system_context})

        # Format history messages
        for message in messages:
            message_content = self._format_message_content(message=message, is_current=False)
            formatted_messages.append(
                {"role": "assistant" if message.role == MessageRole.ASSISTANT else "user", "content": message_content}
            )

        # Add current message
        current_content = self._format_message_content(message=current_message, is_current=True)
        formatted_messages.append({"role": "user", "content": current_content})

        return formatted_messages

    def _format_tools(self, tools: Sequence[MCPTool]) -> list[dict]:
        """
        Format MCP tools for the OpenAI API as function definitions.
        """
        formatted = []
        for tool in tools:
            # Create function parameters schema
            parameters = OpenAIFunctionParameters(
                properties=tool.input_schema.get("properties", {}),
                required=tool.input_schema.get("required", []),
                additionalProperties=False,
            )

            # Create function definition
            function = OpenAIFunction(name=tool.name, description=tool.description or "", parameters=parameters)

            # Create tool wrapper
            tool_def = OpenAITool(function=function)
            formatted.append(tool_def)

        return [tool.model_dump() for tool in formatted]

    def _format_tool_result_for_messages(self, content: list[TextContent | ImageContent | EmbeddedResource]) -> str:
        """
        Format tool result content for message history
        """
        text_parts = []

        for item in content:
            if isinstance(item, TextContent):
                text_parts.append(item.text)
            else:
                raise NotImplementedError("ImageContent and EmbeddedResource not supported yet")
        return " ".join(text_parts)

    async def _should_cancel(self, session_id: UUID | None) -> bool:
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

    async def _execute_tool(self, name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolResult:
        """
        Execute a tool using the MCP service.

        Args:
            name: Name of the tool to execute
            arguments: Tool arguments
            call_id: Optional call identifier for tracking

        Returns:
            ToolResult containing the execution result
        """
        try:
            mcp_service = MCPService()
            tool_call = ToolCall(name=name, arguments=arguments, call_id=call_id)
            return await mcp_service.execute_tool(tool_call=tool_call)
        except Exception as error:
            logger.exception(f"Tool execution failed: {error}")
            raise

    def _handle_completion(self, chunk: Any) -> tuple[tuple[int, int] | None, bool]:
        """
        Handle completion/finish of a stream chunk.

        Args:
            chunk: Response chunk from OpenAI API

        Returns:
            Tuple of (token_usage, should_stop)
            - token_usage: Tuple of (prompt_tokens, completion_tokens) or None
            - should_stop: Boolean indicating if streaming should stop
        """

        finish_reason = chunk.choices[0].finish_reason
        if finish_reason == "stop":
            if chunk.usage:
                self._last_usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                return self._last_usage, True
            return None, True
        return None, False

    async def _accumulate_tool_call(self, delta_tool_call: Any) -> dict[str, Any]:
        """
        Accumulates a tool call from stream chunks
        """
        return {
            "id": delta_tool_call.id,
            "name": delta_tool_call.function.name,
            "arguments": delta_tool_call.function.arguments,
            "type": "function",
        }

    async def generate_stream(
        self,
        current_message: ChatMessage,
        model: str,
        system_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        messages: Sequence[ChatMessage] | None = None,
        session_id: UUID | None = None,
        available_tools: Sequence[MCPTool] | None = None,
    ) -> AsyncGenerator[StreamBlock, None]:
        """
        Generate streaming text using OpenAI with integrated tool calling.
        When a function call is encountered in the stream, execute the tool via MCPService
        and append its result back to the conversation.
        """
        try:
            # Signal initial thinking state
            yield StreamManager.create_thinking_block(content="Thinking...", metadata={"phase": "initialization"})

            formatted_messages = self._prepare_messages(
                messages=messages or [],
                current_message=current_message,
                system_context=system_context,
            )

            # Prepare tools if available
            if available_tools:
                formatted_tools = self._format_tools(tools=available_tools)
            else:
                formatted_tools = None

            stream = await self._client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                tools=formatted_tools,
                tool_choice="auto",
            )

            current_tool_calls: dict[str, dict] = {}
            current_tool_index: dict[int, str] = {}
            current_content = ""
            conversation_messages = formatted_messages.copy()

            async for chunk in stream:
                if await self._should_cancel(session_id):
                    yield StreamManager.create_content_block(
                        content="Request cancelled", metadata={"status": "cancelled"}
                    )
                    break

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle regular content
                if content := getattr(delta, "content", None):
                    current_content += content
                    yield StreamManager.create_content_block(content=content)

                # Handle tool calls
                if tool_calls := getattr(delta, "tool_calls", []):
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
                                yield StreamManager.create_thinking_block(
                                    content=f"Preparing to use tool: {tool_call.function.name}",
                                    metadata={"phase": "tool_preparation"},
                                )
                                yield StreamManager.create_tool_start_block(
                                    tool_name=tool_call.function.name,
                                    tool_call_id=tool_id,
                                    metadata={"status": "starting"},
                                )
                        elif tool_id and tool_call.function.arguments:
                            # Accumulate arguments for existing tool call
                            current_tool_calls[tool_id]["arguments"] += tool_call.function.arguments

                # Handle tool execution
                if chunk.choices[0].finish_reason == "tool_calls":
                    for tool_id, tool_call in current_tool_calls.items():
                        try:
                            parsed_args = json.loads(tool_call["arguments"])

                            yield StreamManager.create_thinking_block(
                                content=f"Executing tool: {tool_call['name']}", metadata={"phase": "tool_execution"}
                            )

                            yield StreamManager.create_tool_call_block(
                                tool_name=tool_call["name"],
                                tool_args=parsed_args,
                                tool_call_id=tool_id,
                                metadata={"status": "executing"},
                            )

                            tool_result = await self._execute_tool(name=tool_call["name"], arguments=parsed_args)

                            # Format tool result for message history
                            formatted_result = self._format_tool_result_for_messages(tool_result.content)

                            # Update messages with tool results
                            conversation_messages.extend(
                                [
                                    {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": tool_call["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call["name"],
                                                    "arguments": tool_call["arguments"],
                                                },
                                            }
                                        ],
                                    },
                                    {"role": "tool", "content": formatted_result, "tool_call_id": tool_id},
                                ]
                            )

                            yield StreamManager.create_tool_result_block(
                                content=tool_result.content,
                                tool_call_id=tool_id,
                                tool_name=tool_call["name"],
                                metadata={
                                    "status": "completed",
                                    "content_types": [item.type for item in tool_result.content],
                                },
                            )

                            # Signal thinking state for processing tool result
                            yield StreamManager.create_thinking_block(
                                content=f"Processing {tool_call['name']} results...",
                                metadata={"phase": "tool_result_processing"},
                            )

                        except Exception as tool_error:
                            yield StreamManager.create_error_block(
                                error_type="tool_execution_error",
                                error_detail=str(tool_error),
                                metadata={"tool_name": tool_call["name"], "tool_id": tool_id},
                            )

                    # Continue stream with tool results
                    yield StreamManager.create_thinking_block(
                        content="Continuing with tool results...", metadata={"phase": "continuation"}
                    )

                    continue_stream = await self._client.chat.completions.create(
                        model=model,
                        messages=conversation_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                        tools=formatted_tools,
                        tool_choice="auto",
                    )

                    async for cont_chunk in continue_stream:
                        if not cont_chunk.choices:
                            continue
                        if content := getattr(cont_chunk.choices[0].delta, "content", None):
                            current_content += content
                            yield StreamManager.create_content_block(content=content)
                        # Use the same completion handling logic
                        usage, should_stop = self._handle_completion(cont_chunk)
                        if should_stop:
                            if usage:
                                self._last_usage = usage
                            yield StreamManager.create_done_block(
                                metadata={"final_content": current_content, "token_usage": self._last_usage}
                            )
                            break

                elif chunk.choices[0].finish_reason == "stop":
                    if not chunk.choices:
                        continue
                    # Handle regular completion
                    usage, should_stop = self._handle_completion(chunk)
                    if should_stop:
                        if usage:
                            self._last_usage = usage
                        yield StreamManager.create_done_block(
                            metadata={"final_content": current_content, "token_usage": self._last_usage}
                        )

        except Exception as error:
            logger.exception("Error in generate_stream")
            yield StreamManager.create_error_block(
                error_type=type(error).__name__, error_detail=str(error), metadata={"phase": "stream_generation"}
            )

    async def generate(
        self,
        current_message: ChatMessage,
        model: str,
        system_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        messages: Sequence[ChatMessage] | None = None,
    ) -> tuple[str, int, int] | None:
        """
        Generate text using OpenAI.
        Args:
            prompt: Input prompt text
            model: Model name to use
            system_context: System context/instructions
            messages: Optional previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        Returns:
            Tuple of (generated text, input tokens, output tokens)
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [],
            current_message=current_message,
            system_context=system_context,
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            generated_text = response.choices[0].message.content or ""
            self._last_usage = (
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

            return (
                generated_text,
                self._last_usage[0],
                self._last_usage[1],
            )

        except (
            APIConnectionError,
            RateLimitError,
            AuthenticationError,
            APIStatusError,
            APIError,
        ) as error:
            self._handle_api_error(error)

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get the token usage from the last operation.
        Returns tuple of (prompt_tokens, completion_tokens)
        """
        return self._last_usage if self._last_usage else (0, 0)


# Register the OpenAI provider with the factory
ProviderFactory.register(provider_type=ProviderType.OPENAI, provider_class=OpenAIProvider)
