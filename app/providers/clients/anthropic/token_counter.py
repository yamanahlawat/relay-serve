from typing import Sequence

from anthropic import Anthropic
from anthropic.types import MessageParam

from app.chat.constants import MessageRole
from app.chat.models import ChatMessage
from app.providers.clients.base import TokenCounterBase
from app.providers.models import LLMProvider


class AnthropicTokenCounter(TokenCounterBase):
    """
    Token counting implementation for Anthropic models
    """

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.client = Anthropic(
            api_key=provider.api_key,
            base_url=provider.base_url or None,
        )
        self.model = model

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in a single text string using Anthropic's API
        """
        response = self.client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model,
            messages=[MessageParam(role="user", content=text)],
        )
        return response.input_tokens

    async def count_message_tokens(self, messages: Sequence[ChatMessage], prompt: str) -> tuple[int, int]:
        """
        Count tokens in message history and new prompt
        Returns (prompt_tokens, context_tokens)
        """
        # Convert messages to Anthropic format
        message_params = []

        # Add previous messages
        for message in messages:
            role = "assistant" if message.role == MessageRole.ASSISTANT else "user"
            message_params.append(MessageParam(role=role, content=message.content))

        # Add new prompt
        message_params.append(MessageParam(role="user", content=prompt))

        # Count tokens
        response = self.client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model,
            messages=message_params,
        )

        # For Anthropic, separate prompt and context tokens
        prompt_tokens = await self.count_tokens(prompt)
        context_tokens = response.input_tokens - prompt_tokens

        return prompt_tokens, context_tokens
