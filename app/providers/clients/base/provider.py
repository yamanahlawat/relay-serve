from abc import ABC, abstractmethod
from typing import AsyncGenerator

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
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate text using the provider.
        Args:
            prompt: The input prompt text.
            model: The name of the model to use for generation.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature parameter for generation.
                Higher values make output more random and creative; lower values
                make output more focused and deterministic.
        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the provider in a streaming manner.
        Args:
            prompt: The input prompt text.
            model: The name of the model to use for generation.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature parameter for generation.
                Higher values make output more random and creative; lower values
                make output more focused and deterministic.
        Yields:
            str: The generated text chunks in a stream.
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_models(cls) -> list[str]:
        """
        Get list of default supported models
        """
        pass
