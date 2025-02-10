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
from app.chat.services.sse import get_sse_manager
from app.files.image.processor import ImageProcessor
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
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate streaming text using OpenAI.
        Args:
            current_message: The current message to generate completion for.
            model: The name of the model to use for generation.
            system_context: The system context to use for generation.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature parameter for generation.
                Higher values make output more random and creative; lower values
                make output more focused and deterministic.
            top_p: Top-p parameter for generation.
                Higher values make output more random and creative; lower values
                make output more focused and deterministic.
            messages: Optional previous conversation messages.
            session_id: Optional session ID for stopping stream.
        Yields:
            Tuple of (chunk text, is_final)
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [],
            current_message=current_message,
            system_context=system_context,
        )
        cancel_key = f"sse:cancel:{session_id}" if session_id else None
        sse_manager = await get_sse_manager()

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                # TODO: In case of o1, o3 models, make max_tokens, temperature, top_p optional
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stream_options={"include_usage": True},
            )

            final_flag = False
            async for chunk in stream:
                if cancel_key and await sse_manager.redis.exists(cancel_key):
                    logger.warning(f"Stream cancelled for session {session_id}")
                    await stream.close()  # Explicitly close the stream from the SDK
                    break

                chunk_choices = chunk.choices
                # Case 1: Chunk with choices present
                if chunk_choices:
                    # If this chunk signals finish ("stop")
                    if chunk_choices[0].finish_reason == "stop":
                        # If usage is already provided in this chunk (e.g. Codestral-2501)
                        if chunk.usage is not None:
                            self._last_usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                            yield ("", True)
                            break
                        else:
                            # Otherwise, set a flag to expect an extra chunk with usage data.
                            final_flag = True
                            continue

                    # Yield content if available.
                    if chunk_choices[0].delta.content:
                        yield (chunk_choices[0].delta.content, False)

                # Case 2: Chunk with no choices but with usage data
                elif final_flag and chunk.usage is not None:
                    self._last_usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                    yield ("", True)
                    break

        except (
            APIConnectionError,
            RateLimitError,
            AuthenticationError,
            APIStatusError,
            APIError,
        ) as error:
            self._handle_api_error(error)

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
