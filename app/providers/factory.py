from typing import Type

from app.providers.clients.base import LLMProviderBase
from app.providers.constants import ProviderType
from app.providers.exceptions import ProviderConfigurationError
from app.providers.models import LLMProvider


class ProviderFactory:
    """
    Factory for creating LLM provider instances.
    """

    _providers: dict[ProviderType, Type[LLMProviderBase]] = {}

    @classmethod
    def register(cls, provider_type: ProviderType, provider_class: Type[LLMProviderBase]) -> None:
        """
        Register a provider implementation for a specific provider type.
        Args:
            provider_type: Type of the provider (OPENAI, ANTHROPIC, Ollama etc.)
            provider_class: Provider implementation class
        """
        cls._providers[provider_type] = provider_class

    @classmethod
    def get_client(cls, provider: LLMProvider) -> LLMProviderBase:
        """
        Get a provider client instance based on provider configuration.
        Args:
            provider: Provider configuration from database
        Returns:
            Configured provider client instance
        Raises:
            ProviderConfigurationError: If provider type not registered
        """
        client_class = cls._providers.get(provider.type)
        if not client_class:
            raise ProviderConfigurationError(
                provider=provider.type, message=f"Provider type {provider.type} not registered"
            )

        return client_class(provider=provider)
