from typing import AsyncGenerator, Sequence

from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from app.chat.constants import MessageRole, llm_defaults
from app.chat.models import ChatMessage
from app.providers.base import LLMProviderBase
from app.providers.constants import OpenAIModelName, ProviderType
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

    def _prepare_messages(
        self,
        messages: Sequence[ChatMessage],
        new_prompt: str,
        system_context: str,
    ) -> list[dict[str, str]]:
        """
        Prepare message history for OpenAI API.
        Args:
            messages: Previous messages in the conversation
            new_prompt: New user prompt to append
            system_context: System context/instructions
        Returns:
            List of formatted messages for the OpenAI API
        """
        formatted_messages = []

        # Add system message if provided
        if system_context:
            formatted_messages.append({"role": "system", "content": system_context})

        # Add conversation history
        for message in messages:
            formatted_messages.append(
                {
                    "role": "assistant" if message.role == MessageRole.ASSISTANT else "user",
                    "content": message.content,
                }
            )

        # Add the new prompt
        formatted_messages.append({"role": "user", "content": new_prompt})
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
            )
        elif isinstance(error, RateLimitError):
            logger.exception("OpenAI API rate limit exceeded during generation")
            raise ProviderRateLimitError(
                provider=self.provider_type,
                error=str(error),
            )
        elif isinstance(error, AuthenticationError):
            logger.exception("OpenAI authentication error during generation")
            raise ProviderConfigurationError(
                provider=self.provider_type,
                message="Invalid API credentials",
                error=str(error),
            )
        elif isinstance(error, APIStatusError):
            logger.exception("OpenAI API status error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=error.status_code,
                message=error.message,
                error=str(error),
            )
        elif isinstance(error, APIError):
            logger.exception("OpenAI API error during generation")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=getattr(error, "status_code", 500),
                error=str(error),
            )

    def _get_usage_from_response(self, response: ChatCompletion) -> tuple[int, int]:
        """
        Extract token usage from OpenAI response.
        """
        usage = response.usage
        return (
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        system_context: str = "",
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = llm_defaults.MAX_TOKENS,
        temperature: float = llm_defaults.TEMPERATURE,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate streaming text using OpenAI.
        Args:
            prompt: Input prompt text
            model: Model name to use
            system_context: System context/instructions
            messages: Optional previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        Yields:
            Tuple of (chunk text, is_final)
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [],
            new_prompt=prompt,
            system_context=system_context,
        )

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            # Track full content for Langfuse
            full_content = ""
            async for chunk in stream:
                if chunk.choices and (content := chunk.choices[0].delta.content):
                    full_content += content
                    yield (content, False)

                # On last chunk, get usage metrics
                if chunk.choices and chunk.choices[0].finish_reason:
                    # Make a non-streaming call to get usage metrics
                    final_response = await self._client.chat.completions.create(
                        model=model,
                        messages=formatted_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    self._last_usage = self._get_usage_from_response(final_response)

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
        max_tokens: int = llm_defaults.MAX_TOKENS,
        temperature: float = llm_defaults.TEMPERATURE,
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
            new_prompt=prompt,
            system_context=system_context,
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            generated_text = response.choices[0].message.content or ""
            self._last_usage = self._get_usage_from_response(response)

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

    @classmethod
    def get_default_models(cls) -> list[str]:
        """
        Get list of default supported models.
        """
        return OpenAIModelName.default_models()

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get the token usage from the last operation.
        Returns tuple of (prompt_tokens, completion_tokens)
        """
        return self._last_usage if self._last_usage else (0, 0)


# Register the OpenAI provider with the factory
ProviderFactory.register(provider_type=ProviderType.OPENAI, provider_class=OpenAIProvider)
