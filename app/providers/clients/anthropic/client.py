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
from anthropic.types import ImageBlockParam, MessageParam
from loguru import logger

from app.chat.constants import AttachmentType, MessageRole
from app.chat.models import ChatMessage
from app.chat.services.sse import get_sse_manager
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
from app.storages.utils import encode_image_to_base64


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

        if current_message.attachments:
            for attachment in current_message.attachments:
                if attachment.type == AttachmentType.IMAGE.value:
                    base64_image = encode_image_to_base64(image_path=attachment.storage_path)
                    image_block = ImageBlockParam(
                        type="image",
                        source={"type": "base64", "media_type": attachment.mime_type, "data": base64_image},
                    )
                    formatted_messages.append(image_block)
        # Append the new prompt as the latest user message.
        formatted_messages.append(MessageParam(role=MessageRole.USER.value, content=current_message.content))
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
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate streaming text using Anthropic Claude.

        Yields:
            A tuple of (text chunk, is_final). When the stream is complete,
            an empty string with is_final=True is yielded.
        """
        message_params = self._prepare_messages(messages=messages or [], current_message=current_message)
        cancel_key = f"sse:cancel:{session_id}" if session_id else None
        sse_manager = await get_sse_manager()

        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=message_params,
                system=system_context,
            ) as stream:
                async for text in stream.text_stream:
                    if cancel_key and await sse_manager.redis.exists(cancel_key):
                        logger.warning(f"Stream cancelled for session {session_id}")
                        await stream.close()  # Close the stream explicitly via the SDK.
                        break
                    # Yield text chunks as they arrive.
                    yield (text, False)

                # When the stream completes, retrieve usage information.
                final_message = await stream.get_final_message()
                self._last_usage = final_message.usage

                # Signal completion with empty text and final flag
                yield ("", True)

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
