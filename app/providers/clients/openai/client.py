import json
from typing import AsyncGenerator, Sequence
from uuid import UUID

from loguru import logger
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
from app.chat.schemas.stream import CompletionMetadata, StreamBlock, ToolExecution
from app.chat.services.stream_block_factory import StreamBlockFactory
from app.files.image.processor import ImageProcessor
from app.model_context_protocol.exceptions import MCPToolError
from app.model_context_protocol.schemas.tools import MCPTool
from app.providers.clients.base import LLMProviderBase
from app.providers.clients.openai.stream import OpenAIStreamHandler
from app.providers.clients.openai.tool import OpenAIToolHandler
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
        self.tool_handler = OpenAIToolHandler()
        self.stream_handler = OpenAIStreamHandler(client=self._client)

    def _handle_api_error(self, error: APIError) -> None:
        """Handle OpenAI API errors and raise appropriate provider exceptions."""
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
        """Format message content with any image attachments."""
        # Add text content
        content = [{"type": "text", "text": message.content}]

        # Add any image attachments
        if is_current and message.attachments:
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ImageProcessor.encode_image_to_base64(attachment.storage_path)}",
                    },
                }
                for attachment in message.direct_attachments
                if attachment.type == AttachmentType.IMAGE.value
            )

        return content

    def _prepare_messages(
        self,
        messages: Sequence[ChatMessage],
        current_message: ChatMessage,
        system_context: str,
    ) -> list[dict[str, str]]:
        """Prepare message history for OpenAI API."""
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
    ) -> AsyncGenerator[tuple[StreamBlock, CompletionMetadata | None], None]:
        """
        Generate streaming text using OpenAI with integrated tool calling.
        """
        try:
            # Initialize completion metadata and content collection
            completion_metadata = CompletionMetadata()
            content_chunks: list[str] = []
            current_tool_calls: dict[str, ToolExecution] = {}

            # Signal initial thinking state
            yield StreamBlockFactory.create_thinking_block(content="Thinking..."), None

            formatted_messages = self._prepare_messages(
                messages=messages or [],
                current_message=current_message,
                system_context=system_context,
            )

            # Format tools if available
            tools_payload = self.tool_handler.format_tools(tools=available_tools) if available_tools else None

            # Initialize conversation messages with the formatted history
            conversation_messages = formatted_messages.copy()

            while True:  # Continue until all tool calls are processed
                stream = await self.stream_handler.create_completion_stream(
                    model=model,
                    messages=conversation_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools_payload,
                )

                current_tool_index: dict[int, str] = {}
                has_tool_calls = False

                async for chunk in stream:
                    if await self.stream_handler.should_cancel(session_id):
                        yield StreamBlockFactory.create_content_block(content="Request cancelled"), None
                        await stream.close()
                        return

                    # Handle regular content
                    if chunk.choices and (content := getattr(chunk.choices[0].delta, "content", None)):
                        content_chunks.append(content)
                        yield StreamBlockFactory.create_content_block(content=content), None

                    # Check for completion/usage
                    usage, should_stop = self.stream_handler.handle_completion(chunk)
                    if usage:
                        self._last_usage = usage
                    if should_stop and not has_tool_calls:
                        completion_metadata.content = "".join(content_chunks)
                        completion_metadata.tool_executions = list(current_tool_calls.values())
                        yield StreamBlockFactory.create_done_block(), completion_metadata
                        return

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Handle tool calls
                    tool_calls, tool_index, tool_event = self.stream_handler.handle_tool_calls(
                        delta=delta,
                        current_tool_calls=current_tool_calls,
                        current_tool_index=current_tool_index,
                    )

                    current_tool_index = tool_index

                    # Handle new tool event
                    if tool_event:
                        tool_execution = ToolExecution(
                            id=tool_event["id"],
                            name=tool_event["name"],
                            arguments=tool_event["arguments"],
                        )
                        current_tool_calls[tool_event["id"]] = tool_execution

                        yield (
                            StreamBlockFactory.create_tool_start_block(
                                tool_name=tool_event["name"], tool_call_id=tool_event["id"]
                            ),
                            None,
                        )

                    # Process tool calls one at a time
                    if chunk.choices[0].finish_reason == "tool_calls":
                        has_tool_calls = True
                        for tool_id, tool_call in tool_calls.items():
                            try:
                                parsed_args = json.loads(tool_call.arguments)

                                # Update tool execution with parsed arguments
                                if tool_id in current_tool_calls:
                                    current_tool_calls[tool_id].arguments = parsed_args

                                yield (
                                    StreamBlockFactory.create_tool_call_block(
                                        tool_name=tool_call.name,
                                        tool_args=parsed_args,
                                        tool_call_id=tool_id,
                                    ),
                                    None,
                                )

                                tool_result = await self.tool_handler.execute_tool(
                                    name=tool_call.name,
                                    arguments=parsed_args,
                                    call_id=tool_id,
                                )

                                formatted_result = self.tool_handler.format_tool_result(tool_result.content)

                                # Update tool execution with result
                                if tool_id in current_tool_calls:
                                    current_tool_calls[tool_id].result = formatted_result

                                # Add this tool's messages to conversation
                                tool_messages = self.tool_handler.format_tool_messages(
                                    tool_call=tool_call,
                                    result=formatted_result,
                                )
                                conversation_messages.extend(tool_messages)

                                yield (
                                    StreamBlockFactory.create_tool_result_block(
                                        content=tool_result.content,
                                        tool_call_id=tool_id,
                                        tool_name=tool_call.name,
                                    ),
                                    None,
                                )

                                yield (
                                    StreamBlockFactory.create_thinking_block(
                                        content=f"Processing {tool_call.name} results..."
                                    ),
                                    None,
                                )

                                break

                            except json.JSONDecodeError as e:
                                error_msg = f"Invalid tool arguments format: {str(e)}"
                                if tool_id in current_tool_calls:
                                    current_tool_calls[tool_id].error = error_msg
                                yield (
                                    StreamBlockFactory.create_error_block(
                                        error_type="tool_argument_error",
                                        error_detail=error_msg,
                                    ),
                                    None,
                                )
                                continue

                            except MCPToolError as e:
                                error_msg = f"Tool execution failed: {str(e)}"
                                if tool_id in current_tool_calls:
                                    current_tool_calls[tool_id].error = error_msg
                                yield (
                                    StreamBlockFactory.create_error_block(
                                        error_type="tool_execution_error",
                                        error_detail=error_msg,
                                    ),
                                    None,
                                )
                                continue

                        break

                if not has_tool_calls:
                    break

            # Final yield with completion metadata
            completion_metadata.content = "".join(content_chunks)
            completion_metadata.tool_executions = list(current_tool_calls.values())
            yield StreamBlockFactory.create_done_block(), completion_metadata

        except Exception as error:
            logger.exception("Error in generate_stream")
            yield (
                StreamBlockFactory.create_error_block(
                    error_type=type(error).__name__,
                    error_detail=str(error),
                ),
                None,
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
        """Generate text using OpenAI."""
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
        """Get token usage from the last operation."""
        return self._last_usage if self._last_usage else (0, 0)


# Register the OpenAI provider with the factory
ProviderFactory.register(provider_type=ProviderType.OPENAI, provider_class=OpenAIProvider)
