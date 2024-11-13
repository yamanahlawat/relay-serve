from typing import AsyncGenerator, Sequence

from anthropic import Anthropic
from anthropic.types import MessageParam

from app.chat.constants import MessageRole
from app.chat.models import ChatMessage
from app.core.config import settings
from app.providers.base import LLMProviderBase
from app.providers.constants import ClaudeModelName
from app.providers.models import LLMProvider


class AnthropicProvider(LLMProviderBase):
    """
    Anthropic Claude provider implementation.
    """

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self._client = Anthropic(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url or None,
        )

    def _prepare_messages(self, messages: Sequence[ChatMessage], new_prompt: str) -> list[MessageParam]:
        """
        Prepare message history for Anthropic API.
        Args:
            messages: Previous messages in the conversation
            new_prompt: New user prompt to append
        Returns:
            List of formatted messages for the Anthropic API
        """
        message_params = []

        # Add previous messages
        for message in messages:
            # Map our roles to Anthropic roles
            role = MessageRole.ASSISTANT.value if message.role == MessageRole.ASSISTANT else MessageRole.USER.value
            message_params.append(MessageParam(role=role, content=message.content))

        # Add the new prompt
        message_params.append(MessageParam(role=MessageRole.USER.value, content=new_prompt))

        return message_params

    async def validate_connection(self) -> bool:
        """
        Validates the connection to Anthropic API.
        Returns:
            bool: True if connection is valid, False otherwise.
        """
        try:
            # Make a minimal API call to verify credentials
            self._client.messages.create(
                model=ClaudeModelName.CLAUDE_3_SONNET.value,
                max_tokens=1,
                messages=[MessageParam(role=MessageRole.USER.value, content="Hi")],
            )
            return True
        except Exception:
            return False

    async def generate_stream(
        self,
        prompt: str,
        model: str,
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
        message_params = self._prepare_messages(messages or [], prompt)

        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=message_params,
        ) as stream:
            for text in stream.text_stream:
                yield (text, False)

            # Get final message with usage info
            final_message = stream.get_final_message()
            self._last_usage = final_message.usage
            yield ("", True)  # Signal completion with token counts

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get the token usage from the last stream
        """
        return (self._last_usage.input_tokens, self._last_usage.output_tokens)

    async def generate(
        self,
        prompt: str,
        model: str,
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = settings.DEFAULT_MAX_TOKENS,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> tuple[str, int, int]:  # Return content and token counts
        """
        Generate text using Anthropic Claude.

        Args:
            prompt: Input prompt text
            model: Model name to use
            messages: Optional previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Tuple of (generated text, input tokens, output tokens)
        """
        message_params = self._prepare_messages(messages or [], prompt)

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=message_params,
        )

        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    @classmethod
    def get_default_models(cls) -> list[str]:
        """
        Get list of default supported models
        """
        return ClaudeModelName.default_models()
