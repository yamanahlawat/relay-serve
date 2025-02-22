import json
from typing import AsyncGenerator, Sequence
from uuid import UUID

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from loguru import logger

from app.chat.constants import AttachmentType, MessageRole
from app.chat.models import ChatMessage
from app.chat.schemas.stream import CompletionMetadata, StreamBlock, ToolExecution
from app.chat.services.sse import get_sse_manager
from app.chat.services.stream_block_factory import StreamBlockFactory
from app.files.image.processor import ImageProcessor
from app.model_context_protocol.schemas.tools import MCPTool
from app.providers.clients.anthropic.tool import AnthropicToolHandler
from app.providers.clients.base import LLMProviderBase
from app.providers.constants import ProviderType
from app.providers.exceptions import (
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from app.providers.factory import ProviderFactory
from app.providers.models import LLMProvider


class AnthropicProvider(LLMProviderBase):
    """
    Anthropic Claude provider implementation with enhanced connection handling.
    """

    # Connection settings
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 60.0  # seconds

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self._last_usage = None
        # Initialize AsyncAnthropic client
        self._client = AsyncAnthropic(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url or None,
            timeout=self.REQUEST_TIMEOUT,
            max_retries=self.MAX_RETRIES,
        )
        self.tool_handler = AnthropicToolHandler()

    def _prepare_messages(self, messages: Sequence[ChatMessage], current_message: ChatMessage) -> list[MessageParam]:
        """
        Prepare message history for the Anthropic API.
        """
        formatted_messages = []

        for message in messages:
            formatted_messages.append(
                MessageParam(
                    role=MessageRole.ASSISTANT.value
                    if message.role == MessageRole.ASSISTANT
                    else MessageRole.USER.value,
                    content=message.content,
                )
            )

        attachments = []
        if current_message.attachments:
            for attachment in current_message.direct_attachments:
                if attachment.type == AttachmentType.IMAGE.value:
                    base64_image = ImageProcessor.encode_image_to_base64(image_path=attachment.storage_path)
                    image_block = ImageBlockParam(
                        type="image",
                        source={"type": "base64", "media_type": attachment.mime_type, "data": base64_image},
                    )
                    attachments.append(image_block)
        # Append the new prompt as the latest user message.
        text_block = TextBlockParam(type="text", text=current_message.content)
        formatted_messages.append(MessageParam(role=MessageRole.USER.value, content=[text_block, *attachments]))
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
        Generate streaming response with tool support.
        Enhanced with improved text accumulation and stream block management.
        """
        try:
            # Signal initial thinking state
            yield StreamBlockFactory.create_thinking_block(), None

            # Initialize completion metadata and content collection
            completion_metadata = CompletionMetadata()
            content_chunks: list[str] = []
            current_tool_calls: dict[str, ToolExecution] = {}
            stream_blocks: list[StreamBlock] = []

            # Format messages for Anthropic API
            message_params = self._prepare_messages(messages=messages or [], current_message=current_message)

            # Format tools if available
            tools_payload = self.tool_handler.format_tools(tools=available_tools) if available_tools else []

            # Initialize conversation with formatted history
            conversation_messages = message_params.copy()

            while True:  # Continue until we get a final response
                async with self._client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    messages=conversation_messages,
                    system=system_context,
                    tools=tools_payload,
                ) as stream:
                    tool_id = None
                    tool_name = None
                    current_partial_json = ""

                    async for chunk in stream:
                        # Check for cancellation
                        if session_id:
                            sse_manager = await get_sse_manager()
                            cancel_key = f"sse:cancel:{session_id}"
                            if await sse_manager.redis.exists(cancel_key):
                                yield (
                                    StreamBlockFactory.create_error_block(
                                        error_type="request_cancelled", error_detail="Stream cancelled by user"
                                    ),
                                    None,
                                )
                                await stream.close()
                                return

                        # Handle chunk based on type
                        if chunk.type == "message_start":
                            if hasattr(chunk.message, "usage"):
                                self._last_usage = chunk.message.usage

                        elif chunk.type == "content_block_start":
                            if chunk.content_block.type == "tool_use":
                                # Store any accumulated content before tool execution
                                if content_chunks:
                                    block = StreamBlockFactory.create_content_block(content="".join(content_chunks))
                                    stream_blocks.append(block)
                                    content_chunks = []

                                tool_id = chunk.content_block.id
                                tool_name = chunk.content_block.name
                                current_tool_calls[tool_id] = ToolExecution(
                                    id=tool_id,
                                    name=tool_name,
                                    arguments={},
                                )
                                block = StreamBlockFactory.create_tool_start_block(
                                    tool_name=tool_name,
                                    tool_call_id=tool_id,
                                )
                                stream_blocks.append(block)
                                yield block, None

                        elif chunk.type == "content_block_delta":
                            if chunk.delta.type == "text_delta":
                                content_chunks.append(chunk.delta.text)
                                block = StreamBlockFactory.create_content_block(content=chunk.delta.text)
                                yield block, None
                            elif chunk.delta.type == "input_json_delta":
                                current_partial_json += chunk.delta.partial_json

                        elif chunk.type == "content_block_stop":
                            if chunk.content_block.type == "tool_use":
                                try:
                                    # Parse accumulated JSON
                                    tool_input = json.loads(current_partial_json)
                                    if tool_id in current_tool_calls:
                                        current_tool_calls[tool_id].arguments = tool_input

                                    # Signal tool call with arguments
                                    block = StreamBlockFactory.create_tool_call_block(
                                        tool_name=tool_name,
                                        tool_args=tool_input,
                                        tool_call_id=tool_id,
                                    )
                                    stream_blocks.append(block)
                                    yield block, None

                                    # Execute the tool
                                    tool_result = await self.tool_handler.execute_tool(
                                        name=tool_name,
                                        arguments=tool_input,
                                        call_id=tool_id,
                                    )

                                    # Format and store result
                                    formatted_result = self.tool_handler.format_tool_result(tool_result.content)
                                    current_tool_calls[tool_id].result = formatted_result

                                    # Add tool messages to conversation
                                    tool_messages = self.tool_handler.format_tool_messages(
                                        tool_name=tool_name,
                                        tool_args=tool_input,
                                        tool_result=formatted_result,
                                        tool_id=tool_id,
                                    )
                                    conversation_messages.extend(tool_messages)

                                    # Signal tool result
                                    block = StreamBlockFactory.create_tool_result_block(
                                        tool_result=tool_result.content,
                                        tool_call_id=tool_id,
                                        tool_name=tool_name,
                                    )
                                    stream_blocks.append(block)
                                    yield block, None

                                    yield (
                                        StreamBlockFactory.create_thinking_block(
                                            content=f"Processing {tool_name} results..."
                                        ),
                                        None,
                                    )

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

                        elif chunk.type == "message_delta":
                            if hasattr(chunk.delta, "usage"):
                                self._last_usage = chunk.usage

                        elif chunk.type == "message_stop":
                            if chunk.message.stop_reason == "tool_use" and tool_id and tool_name:
                                # Continue with a new stream iteration
                                break

                            elif chunk.message.stop_reason == "end_turn":
                                # Store any remaining content
                                if content_chunks:
                                    block = StreamBlockFactory.create_content_block(content="".join(content_chunks))
                                    stream_blocks.append(block)

                                # Final yield with completion metadata
                                completion_metadata.content = "".join(content_chunks)
                                completion_metadata.stream_blocks = stream_blocks
                                yield StreamBlockFactory.create_done_block(), completion_metadata
                                return

                # If we're here, we've executed a tool and need to continue with the next stream iteration
                continue

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
        """
        Generate text using Anthropic Claude.

        Returns:
            A tuple of (generated text, input tokens, output tokens).
        """
        try:
            message_params = self._prepare_messages(messages=messages or [], current_message=current_message)
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=message_params,
                system=system_context,
            )

            self._last_usage = response.usage
            generated_text = response.content[0].text if response.content else ""

            return (
                generated_text,
                self._last_usage.input_tokens,
                self._last_usage.output_tokens,
            )
        except (
            APIConnectionError,
            RateLimitError,
            AuthenticationError,
            APIStatusError,
            APIError,
        ) as error:
            self._handle_api_error(error)

    def _handle_api_error(self, error: APIError) -> None:
        """
        Handle Anthropic API errors and raise the corresponding provider exceptions.
        Exception chaining (using 'from error') is used to preserve the original traceback.
        """
        if isinstance(error, APIConnectionError):
            logger.exception("Anthropic API connection error during generation")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error=str(error),
            ) from error

        elif isinstance(error, RateLimitError):
            logger.exception("Anthropic API rate limit exceeded during generation")
            raise ProviderRateLimitError(
                provider=self.provider_type,
                error=str(error),
            ) from error

        elif isinstance(error, AuthenticationError):
            logger.exception("Anthropic authentication error during generation")
            raise ProviderConfigurationError(
                provider=self.provider_type,
                message="Invalid API credentials",
                error=str(error),
            ) from error

        elif isinstance(error, APIStatusError):
            logger.exception("Anthropic API status error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=error.status_code,
                message=error.message,
                error=str(error),
            ) from error

        else:
            logger.exception("Anthropic API error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=getattr(error, "status_code", 500),
                error=str(error),
            ) from error

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get the token usage from the last operation.
        """
        if self._last_usage:
            return (
                self._last_usage.input_tokens,
                self._last_usage.output_tokens,
            )
        return (0, 0)


# Register the Anthropic provider with the factory.
ProviderFactory.register(provider_type=ProviderType.ANTHROPIC, provider_class=AnthropicProvider)
