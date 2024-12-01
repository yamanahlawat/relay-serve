from typing import Sequence

import tiktoken
from openai import AsyncOpenAI

from app.chat.models import ChatMessage
from app.providers.clients.base import TokenCounterBase
from app.providers.models import LLMProvider


class OpenAITokenCounter(TokenCounterBase):
    """
    Token counting implementation for OpenAI models using tiktoken
    """

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url or None,
        )
        self.model = model
        # Get the encoding for the model
        self.encoding = tiktoken.encoding_for_model(model)

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in a single text string using tiktoken
        """
        return len(self.encoding.encode(text))

    async def count_message_tokens(self, messages: Sequence[ChatMessage], prompt: str) -> tuple[int, int]:
        """
        Count tokens in message history and new prompt using tiktoken
        Returns (prompt_tokens, context_tokens)
        """
        # Count tokens in the prompt
        prompt_tokens = await self.count_tokens(prompt)

        # Count tokens in the context (previous messages)
        context_tokens = 0
        for message in messages:
            # Add message content tokens
            context_tokens += await self.count_tokens(message.content)

            # Add message metadata tokens (3 for role, content)
            context_tokens += 3

        # Add general message formatting tokens (2 per message)
        context_tokens += 2 * len(messages)

        return prompt_tokens, context_tokens
