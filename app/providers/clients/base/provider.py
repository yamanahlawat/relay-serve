from abc import ABC, abstractmethod
from typing import AsyncGenerator, Sequence
from uuid import UUID

from app.chat.models.message import ChatMessage
from app.providers.constants import ProviderType
from app.providers.models import LLMProvider


class LLMProviderBase(ABC):
    """
    Base class for LLM providers.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.provider_type = ProviderType(provider.type)

    @abstractmethod
    async def generate(
        self,
        current_message: ChatMessage,
        model: str,
        system_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        messages: Sequence[ChatMessage] | None = None,
        session_id: UUID | None = None,
    ) -> str:
        """
        Generate text using the provider.
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
        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
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
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the provider in a streaming manner.
        Args:
            current_message: The current message to generate completion for.
            model: The name of the model to use for generation.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature parameter for generation.
                Higher values make output more random and creative; lower values
                make output more focused and deterministic.
        Yields:
            str: The generated text chunks in a stream.
        """
        pass
