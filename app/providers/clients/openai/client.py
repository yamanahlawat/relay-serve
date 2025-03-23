import json
from typing import Any, AsyncGenerator, Sequence
from uuid import UUID

from llm_registry import CapabilityRegistry, ModelCapabilities, ModelNotFoundError
from loguru import logger
from openai import (
    AsyncOpenAI,
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

    def _format_message_content(
        self, message: ChatMessage, is_current: bool = False, model_capabilities: ModelCapabilities | None = None
    ) -> str | None | list[dict[str, Any]]:
        """
        Format message content with any image attachments based on model capabilities.
        """
        # Add text content
        content = [{"type": "text", "text": message.content}]

        # For models without vision support or unknown models with text-only content, return simple string
        if (model_capabilities and not model_capabilities.features.vision) or (
            not model_capabilities and not message.attachments
        ):
            return content

        # Add any image attachments if present and either model supports vision or capabilities unknown
        if is_current and message.attachments:
            if model_capabilities and not model_capabilities.features.vision:
                logger.warning("Model does not support vision, skipping image attachments")
            else:
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
        model: str,
        model_capabilities: ModelCapabilities | None = None,
    ) -> list[dict[str, str | list[dict]]]:
        """
        Prepare message history for OpenAI API.
        """
        formatted_messages = []

        # Handle system prompt based on model capabilities
        if system_context:
            if model_capabilities and not model_capabilities.features.system_prompt:
                logger.warning(f"Model {model} does not support system prompt, adding as user message")
                formatted_messages.append({"role": "user", "content": system_context})
            else:
                # If capabilities unknown or system prompts supported, use system message
                formatted_messages.append({"role": "system", "content": system_context})

        # Format history messages
        for message in messages:
            message_content = self._format_message_content(
                message=message,
                is_current=False,
                model_capabilities=model_capabilities,
            )
            formatted_messages.append(
                {"role": "assistant" if message.role == MessageRole.ASSISTANT else "user", "content": message_content}
            )

        # Add current message
        current_content = self._format_message_content(
            message=current_message, is_current=True, model_capabilities=model_capabilities
        )
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
            # Signal initial thinking state
            yield StreamBlockFactory.create_thinking_block(), None

            # Initialize completion metadata and content collection
            completion_metadata = CompletionMetadata()
            content_chunks: list[str] = []
            current_tool_calls: dict[str, ToolExecution] = {}
            stream_blocks: list[StreamBlock] = []

            # Track processed tool calls to prevent duplicates
            processed_tool_ids: set[str] = set()

            # Check model capabilities from llm-registry
            model_registry = CapabilityRegistry()
            try:
                model_capabilities = model_registry.get_model(model_id=model.lower())
            except ModelNotFoundError:
                model_capabilities = None

            formatted_messages = self._prepare_messages(
                messages=messages or [],
                current_message=current_message,
                system_context=system_context,
                model=model,
                model_capabilities=model_capabilities,
            )

            # Initialize conversation messages with the formatted history
            conversation_messages = formatted_messages.copy()

            # Build API parameters based on model capabilities
            api_params = {
                "model": model,
                "messages": conversation_messages,
            }

            if model_capabilities:
                # Only include parameters that are supported by the model
                if model_capabilities.api_params.max_tokens:
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    logger.warning(f"Model {model} does not support max_tokens")

                if model_capabilities.api_params.temperature:
                    api_params["temperature"] = temperature
                else:
                    logger.warning(f"Model {model} does not support temperature")

                if model_capabilities.api_params.top_p:
                    api_params["top_p"] = top_p
                else:
                    logger.warning(f"Model {model} does not support top_p")

                if model_capabilities.api_params.stream:
                    api_params["stream"] = True
                else:
                    logger.warning(f"Model {model} does not support streaming")

                # Handle tool-related parameters
                if available_tools and model_capabilities.features.tools:
                    tools_payload = self.tool_handler.format_tools(tools=available_tools)
                    api_params["tools"] = tools_payload
                    api_params["tool_choice"] = "auto"
                else:
                    logger.warning(f"Model {model} does not support tools, skipping tool-related parameters")
            else:
                logger.warning(f"Model {model} capabilities not found in registry, passing all parameters")
                # If model not in registry, pass all parameters
                api_params.update(
                    {
                        "max_completion_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stream": True,
                    }
                )
                # For unknown models, include tools if provided but warn
                if available_tools:
                    logger.warning(f"Model {model} capabilities not found in registry, including tool parameters")
                    api_params["tools"] = self.tool_handler.format_tools(tools=available_tools)
                    api_params["tool_choice"] = "auto"

            has_next_tool_calls = True
            while has_next_tool_calls:  # Continue until no more tool calls are needed
                stream = await self.stream_handler.create_completion_stream(**api_params)
                current_tool_index: dict[int, str] = {}
                has_next_tool_calls = False  # Reset for this iteration
                pending_tool_messages = []  # Store tool messages to add after processing

                async for chunk in stream:
                    if await self.stream_handler.should_cancel(session_id):
                        yield (
                            StreamBlockFactory.create_error_block(
                                error_type="request_cancelled",
                                error_detail="Streaming cancelled by User",
                            ),
                            None,
                        )
                        await stream.aclose()
                        return

                    # Handle regular content
                    if chunk.choices and (content := getattr(chunk.choices[0].delta, "content", None)):
                        content_chunks.append(content)
                        yield StreamBlockFactory.create_content_block(content=content), None

                    # Check for completion/usage
                    usage, should_stop = self.stream_handler.handle_completion(chunk)
                    if usage:
                        self._last_usage = usage
                    if should_stop and not has_next_tool_calls:
                        # Store any remaining content as final block
                        if content_chunks:
                            block = StreamBlockFactory.create_content_block(content="".join(content_chunks))
                            stream_blocks.append(block)

                        completion_metadata.content = "".join(content_chunks)
                        completion_metadata.stream_blocks = stream_blocks
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
                    if tool_event and tool_event["id"] not in processed_tool_ids:
                        # Store accumulated content before tool execution
                        if content_chunks:
                            block = StreamBlockFactory.create_content_block(content="".join(content_chunks))
                            stream_blocks.append(block)
                            content_chunks = []

                        tool_execution = ToolExecution(
                            id=tool_event["id"],
                            name=tool_event["name"],
                            arguments=tool_event["arguments"],
                        )
                        current_tool_calls[tool_event["id"]] = tool_execution

                        block = StreamBlockFactory.create_tool_start_block(
                            tool_name=tool_event["name"],
                            tool_call_id=tool_event["id"],
                        )
                        stream_blocks.append(block)
                        yield block, None

                    # Process tool calls one at a time
                    if chunk.choices[0].finish_reason == "tool_calls":
                        has_next_tool_calls = True
                        for tool_id, tool_call in tool_calls.items():
                            # Skip already processed tool calls
                            if tool_id in processed_tool_ids:
                                continue

                            try:
                                parsed_args = (
                                    tool_call.arguments
                                    if isinstance(tool_call.arguments, dict)
                                    else json.loads(tool_call.arguments)
                                )

                                # Update tool execution with parsed arguments
                                if tool_id in current_tool_calls:
                                    current_tool_calls[tool_id].arguments = parsed_args

                                block = StreamBlockFactory.create_tool_call_block(
                                    tool_name=tool_call.name,
                                    tool_args=parsed_args,
                                    tool_call_id=tool_id,
                                )
                                stream_blocks.append(block)
                                yield block, None

                                # Execute the tool call
                                tool_result = await self.tool_handler.execute_tool(
                                    name=tool_call.name,
                                    arguments=parsed_args,
                                    call_id=tool_id,
                                )

                                formatted_result = self.tool_handler.format_tool_result(tool_result.content)

                                # Update tool execution with result
                                if tool_id in current_tool_calls:
                                    current_tool_calls[tool_id].result = formatted_result

                                # Prepare tool messages to add to conversation after this loop
                                tool_messages = self.tool_handler.format_tool_messages(
                                    tool_call=tool_call,
                                    result=formatted_result,
                                )
                                pending_tool_messages.extend(tool_messages)

                                block = StreamBlockFactory.create_tool_result_block(
                                    tool_result=tool_result.content,
                                    tool_call_id=tool_id,
                                    tool_name=tool_call.name,
                                )
                                stream_blocks.append(block)
                                yield block, None

                                block = StreamBlockFactory.create_thinking_block(
                                    content=f"Processing {tool_call.name} results..."
                                )
                                yield block, None

                                # Mark this tool as processed
                                processed_tool_ids.add(tool_id)

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

                        # Only update conversation messages after processing all tools in this batch
                        if pending_tool_messages:
                            conversation_messages.extend(pending_tool_messages)
                            api_params["messages"] = conversation_messages

                        # Break inner loop to get a new stream with updated messages
                        break

                # If no more tool calls detected, exit the outer loop
                if not has_next_tool_calls:
                    break

            # Final yield with completion metadata
            if content_chunks:
                block = StreamBlockFactory.create_content_block(content="".join(content_chunks))
                stream_blocks.append(block)

            completion_metadata.content = "".join(content_chunks)
            completion_metadata.stream_blocks = stream_blocks
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

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get token usage from the last operation.
        """
        return self._last_usage if self._last_usage else (0, 0)


# Register the OpenAI provider with the factory
ProviderFactory.register(provider_type=ProviderType.OPENAI, provider_class=OpenAIProvider)
