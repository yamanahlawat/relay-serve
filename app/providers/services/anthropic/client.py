from typing import AsyncGenerator, Sequence

import httpx
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)
from anthropic.types import MessageParam
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.chat.constants import MessageRole
from app.chat.models import ChatMessage
from app.core.config import settings
from app.providers.base import LLMProviderBase
from app.providers.constants import ClaudeModelName, ProviderType
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
    BACKOFF_MIN = 1  # seconds
    BACKOFF_MAX = 10  # seconds

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)

        # Initialize aiohttp session with connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.REQUEST_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=100,
                keepalive_expiry=300,
            ),
        )
        # Initialize Anthropic client with retry settings
        self._client = AsyncAnthropic(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url or None,
            timeout=self.REQUEST_TIMEOUT,
            max_retries=self.MAX_RETRIES,
            http_client=self._http_client,
        )
        self._last_usage = None

    async def close(self):
        """Close the http client."""
        if self._http_client:
            await self._http_client.aclose()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=BACKOFF_MIN, max=BACKOFF_MAX),
        reraise=True,
    )
    async def _make_request(self, func, *args, **kwargs):
        """
        Make a request with retry logic and proper error handling.
        """
        try:
            return await func(*args, **kwargs)
        except APIConnectionError as error:
            logger.error(f"Connection error during request: {error}")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error="Failed to connect to Anthropic API",
            ) from error
        except RateLimitError as error:
            logger.warning(f"Rate limit exceeded: {error}")
            raise ProviderRateLimitError(
                provider=self.provider_type,
                error="Anthropic API rate limit exceeded",
            ) from error
        except AuthenticationError as error:
            logger.error(f"Authentication error: {error}")
            raise ProviderConfigurationError(
                provider=self.provider_type,
                error="Invalid Anthropic API credentials",
            ) from error
        except APIStatusError as error:
            logger.error(f"API status error: {error}")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=error.status_code,
                error=f"Anthropic API error: {error}",
            ) from error
        except APIError as error:
            logger.error(f"General API error: {error}")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=getattr(error, "status_code", 500),
                error=f"Unexpected Anthropic API error: {error}",
            ) from error
        except Exception as error:
            logger.error(f"Unexpected error during request: {error}")
            raise ProviderAPIError(
                provider=self.provider_type, status_code=500, error=f"Unexpected error: {error}"
            ) from error

    def _prepare_messages(self, messages: Sequence[ChatMessage], new_prompt: str) -> list[MessageParam]:
        """
        Prepare message history for Anthropic API.
        Args:
            messages: Previous messages in the conversation
            new_prompt: New user prompt to append
        Returns:
            List of formatted messages for the Anthropic API
        """
        message_params = [
            MessageParam(
                role=MessageRole.ASSISTANT.value if message.role == MessageRole.ASSISTANT else MessageRole.USER.value,
                content=message.content,
            )
            for message in messages
        ]
        # Add the new prompt
        message_params.append(MessageParam(role=MessageRole.USER.value, content=new_prompt))
        return message_params

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        system_context: str = "",
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = settings.DEFAULT_MAX_TOKENS,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate streaming text using Anthropic Claude.
        Args:
            prompt: Input prompt text
            model: Model name to use
            messages: Optional previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        Yields:
            Tuple of (chunk text, is_final)
        """
        message_params = self._prepare_messages(messages=messages or [], new_prompt=prompt)

        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=message_params,
                system=system_context,
            ) as stream:
                async for text in stream.text_stream:
                    # Not final chunk
                    yield (text, False)

                # After stream is complete, get final message with usage info
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
        prompt: str,
        model: str,
        system_context: str = "",
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = settings.DEFAULT_MAX_TOKENS,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> tuple[str, int, int]:
        """
        Generate text using Anthropic Claude with improved error handling and retries.
        """
        message_params = self._prepare_messages(messages=messages or [], new_prompt=prompt)

        response = await self._make_request(
            self._client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
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

    def _handle_api_error(self, error: APIError) -> None:
        """
        Handle Anthropic API errors and raise appropriate provider exceptions.
        """
        if isinstance(error, APIConnectionError):
            logger.exception("Anthropic API connection error during generation")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error=str(error),
            )
        elif isinstance(error, RateLimitError):
            logger.exception("Anthropic API rate limit exceeded during generation")
            raise ProviderRateLimitError(
                provider=self.provider_type,
                error=str(error),
            )
        elif isinstance(error, AuthenticationError):
            logger.exception("Anthropic authentication error during generation")
            raise ProviderConfigurationError(
                provider=self.provider_type,
                message="Invalid API credentials",
                error=str(error),
            )
        elif isinstance(error, APIStatusError):
            logger.exception("Anthropic API status error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=error.status_code,
                message=error.message,
                error=str(error),
            )
        elif isinstance(error, APIError):
            logger.exception("Anthropic API error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=getattr(error, "status_code", 500),
                error=str(error),
            )

    @classmethod
    def get_default_models(cls) -> list[str]:
        """
        Get list of default supported models.
        """
        return ClaudeModelName.default_models()

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


# Register the Anthropic provider with the factory
ProviderFactory.register(provider_type=ProviderType.ANTHROPIC, provider_class=AnthropicProvider)
