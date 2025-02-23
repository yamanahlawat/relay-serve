from datetime import datetime
from typing import AsyncGenerator, Sequence
from uuid import UUID, uuid4

from loguru import logger
from ollama import AsyncClient, Image, Message, ResponseError

from app.chat.constants import AttachmentType, MessageRole
from app.chat.models import ChatMessage
from app.chat.schemas.stream import CompletionMetadata, StreamBlock
from app.chat.services.sse import get_sse_manager
from app.chat.services.stream_block_factory import StreamBlockFactory
from app.model_context_protocol.schemas.tools import MCPTool
from app.providers.clients.base import LLMProviderBase
from app.providers.clients.ollama.tool import OllamaToolHandler
from app.providers.constants import ProviderType
from app.providers.exceptions import (
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from app.providers.factory import ProviderFactory
from app.providers.models import LLMProvider


class OllamaProvider(LLMProviderBase):
    """
    Ollama provider implementation with tool call support.
    """

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self._client = AsyncClient(host=self.provider.base_url)
        self._last_usage: tuple[int, int] | None = None
        self.tool_handler = OllamaToolHandler()

    def _get_usage_metrics(self, response: dict) -> dict:
        """
        Extract usage metrics from the Ollama response.
        Args:
            response: Raw response from the Ollama API.
        Returns:
            A dictionary containing usage metrics.
        """
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_duration": response.get("total_duration", 0),  # in nanoseconds
            "load_duration": response.get("load_duration", 0),  # in nanoseconds
            "eval_duration": response.get("eval_duration", 0),  # in nanoseconds
            "created_at": datetime.fromisoformat(response["created_at"].replace("Z", "+00:00")),
            "done_reason": response.get("done_reason"),
        }

    def _prepare_messages(
        self,
        messages: Sequence[ChatMessage],
        current_message: ChatMessage,
        system_context: str,
    ) -> list[Message]:
        """
        Prepare message history for the Ollama API.
        Args:
            messages: Previous messages in the conversation.
            current_message: Current message to process.
            system_context: System context to send as a system message.
        Returns:
            A list of formatted messages for the Ollama API.
        """
        formatted_messages = [Message(role="system", content=system_context)]

        # Format history messages
        for message in messages:
            role = "assistant" if message.role == MessageRole.ASSISTANT else "user"
            formatted_messages.append(Message(role=role, content=message.content))

        # Format current message with any attachments
        message_images = []
        for attachment in current_message.direct_attachments:
            if attachment.type == AttachmentType.IMAGE.value:
                message_images.append(Image(value=attachment.storage_path))

        formatted_messages.append(
            Message(
                role="user",
                content=current_message.content,
                images=message_images,
            )
        )
        return formatted_messages

    async def _handle_api_error(self, error: Exception) -> None:
        """
        Handle Ollama API errors and raise appropriate provider exceptions.
        """
        if isinstance(error, ResponseError):
            if error.status_code == 404:
                logger.exception("Ollama model not found")
                raise ProviderConfigurationError(
                    provider=self.provider_type,
                    message="Model not found or not loaded",
                    error=str(error),
                ) from error
            elif error.status_code == 429:
                logger.exception("Ollama API rate limit exceeded")
                raise ProviderRateLimitError(
                    provider=self.provider_type,
                    error=str(error),
                ) from error
            elif error.status_code >= 500:
                logger.exception("Ollama server error")
                raise ProviderConnectionError(
                    provider=self.provider_type,
                    message="Ollama server error",
                    error=str(error),
                ) from error
            else:
                logger.exception("Ollama API error")
                raise ProviderAPIError(
                    provider=self.provider_type,
                    status_code=error.status_code,
                    error=error.error,
                ) from error
        elif isinstance(error, ConnectionError):
            logger.exception("Ollama connection error")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error=str(error),
            ) from error
        elif isinstance(error, TimeoutError):
            logger.exception("Ollama request timed out")
            raise ProviderConnectionError(
                provider=self.provider_type,
                message="Request timed out",
                error=str(error),
            ) from error
        else:
            logger.exception("Unexpected Ollama error")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=500,
                error=str(error),
            ) from error

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
        Generate streaming text using Ollama with tool support.
        """
        try:
            yield StreamBlockFactory.create_thinking_block(), None

            completion_metadata = CompletionMetadata()
            content_chunks: list[str] = []
            stream_blocks: list[StreamBlock] = []
            conversation_messages = []

            # Format initial messages
            formatted_messages = self._prepare_messages(
                messages=messages or [],
                current_message=current_message,
                system_context=system_context,
            )
            conversation_messages.extend(formatted_messages)

            # Format tools if available
            tools_payload = self.tool_handler.format_tools(tools=available_tools) if available_tools else None

            options = {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            while True:
                stream = await self._client.chat(
                    model=model,
                    messages=conversation_messages,
                    stream=True,
                    options=options,
                    tools=tools_payload,
                )

                has_tool_calls = False
                pending_tool_call = None

                async for chunk in stream:
                    # Check for cancellation
                    if session_id:
                        sse_manager = await get_sse_manager()
                        cancel_key = f"sse:cancel:{session_id}"
                        if await sse_manager.redis.exists(cancel_key):
                            yield (
                                StreamBlockFactory.create_error_block(
                                    error_type="request_cancelled",
                                    error_detail="Stream cancelled by User",
                                ),
                                None,
                            )
                            await stream.aclose()
                            return

                    message = chunk.get("message", {})

                    # Handle regular content
                    if content := message.get("content", ""):
                        content_chunks.append(content)
                        yield StreamBlockFactory.create_content_block(content=content), None

                    # Handle tool calls
                    if tool_calls := message.get("tool_calls", []):
                        has_tool_calls = True

                        # There should be only one tool call per response based on Ollama's behavior
                        tool_call = tool_calls[0]
                        function = tool_call.function

                        # Generate a unique ID for this tool call
                        tool_id = uuid4().hex
                        tool_name = function.name
                        tool_args = function.arguments

                        pending_tool_call = {"id": tool_id, "name": tool_name, "args": tool_args}

                    # Process tool call at the end of the response
                    if chunk.get("done", False):
                        if pending_tool_call:
                            tool_id = pending_tool_call["id"]
                            tool_name = pending_tool_call["name"]
                            tool_args = pending_tool_call["args"]

                            try:
                                # Signal tool start
                                block = StreamBlockFactory.create_tool_start_block(
                                    tool_name=tool_name,
                                    tool_call_id=tool_id,
                                )
                                stream_blocks.append(block)
                                yield block, None

                                # Signal tool call
                                block = StreamBlockFactory.create_tool_call_block(
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tool_call_id=tool_id,
                                )
                                stream_blocks.append(block)
                                yield block, None

                                # Execute tool
                                tool_result = await self.tool_handler.execute_tool(
                                    name=tool_name,
                                    arguments=tool_args,
                                    call_id=tool_id,
                                )

                                # Format result
                                formatted_result = self.tool_handler.format_tool_result(tool_result.content)

                                # Add tool response to conversation
                                conversation_messages.append(
                                    {"role": "tool", "content": formatted_result, "name": tool_name}
                                )

                                # Signal tool result
                                block = StreamBlockFactory.create_tool_result_block(
                                    tool_result=tool_result.content,
                                    tool_call_id=tool_id,
                                    tool_name=tool_name,
                                )
                                stream_blocks.append(block)
                                yield block, None

                                # Continue processing with thinking state
                                yield (
                                    StreamBlockFactory.create_thinking_block(
                                        content=f"Processing {tool_name} results..."
                                    ),
                                    None,
                                )

                            except Exception as e:
                                error_msg = f"Tool execution failed: {str(e)}"
                                yield (
                                    StreamBlockFactory.create_error_block(
                                        error_type="tool_execution_error",
                                        error_detail=error_msg,
                                    ),
                                    None,
                                )
                                return

                        else:
                            # No tool calls, just regular completion
                            if content_chunks:
                                block = StreamBlockFactory.create_content_block(content="".join(content_chunks))
                                stream_blocks.append(block)

                            # Update usage metrics
                            metrics = self._get_usage_metrics(response=chunk)
                            self._last_usage = (metrics["prompt_tokens"], metrics["completion_tokens"])

                            # Final yield with completion metadata
                            completion_metadata.content = "".join(content_chunks)
                            completion_metadata.stream_blocks = stream_blocks
                            yield StreamBlockFactory.create_done_block(), completion_metadata
                            return

                # Continue to next iteration if we had a tool call
                if not has_tool_calls:
                    break

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
        Generate text using Ollama.
        Returns:
            A tuple containing (generated text, input tokens, output tokens).
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [], current_message=current_message, system_context=system_context
        )
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            response = await self._client.chat(
                model=model,
                messages=formatted_messages,
                options=options,
            )
            generated_text = response.get("message", {}).get("content", "")
            metrics = self._get_usage_metrics(response=response)
            # Store usage metrics
            self._last_usage = (metrics["prompt_tokens"], metrics["completion_tokens"])
            return (
                generated_text,
                metrics["prompt_tokens"],
                metrics["completion_tokens"],
            )
        except Exception as error:
            await self._handle_api_error(error)

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get token usage from the last operation.
        Returns:
            A tuple (prompt_tokens, completion_tokens).
        """
        return self._last_usage if self._last_usage is not None else (0, 0)

    async def list_existing_models(self) -> list[str]:
        """
        Fetch available models from the Ollama API.
        Returns:
            A list of model names available in the Ollama instance.
        """
        response = await self._client.list()
        return [model["name"] for model in response.get("models", [])]


# Register the Ollama provider with the factory.
ProviderFactory.register(provider_type=ProviderType.OLLAMA, provider_class=OllamaProvider)
