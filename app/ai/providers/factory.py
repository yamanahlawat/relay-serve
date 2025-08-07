"""Pydantic AI provider factory and management."""

from typing import Any

from pydantic_ai import Agent

from app.ai.providers import (
    AnthropicProviderBuilder,
    BedrockProviderBuilder,
    CohereProviderBuilder,
    GeminiProviderBuilder,
    GroqProviderBuilder,
    MistralProviderBuilder,
    OpenAIProviderBuilder,
    ProviderBuilder,
)
from app.llms.constants import ProviderType
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class ProviderFactory:
    """Factory for creating pydantic-ai models and agents with provider-specific configurations."""

    # Registry of provider builders
    _builders: dict[ProviderType, type[ProviderBuilder]] = {
        ProviderType.OPENAI: OpenAIProviderBuilder,
        ProviderType.ANTHROPIC: AnthropicProviderBuilder,
        ProviderType.GEMINI: GeminiProviderBuilder,
        ProviderType.GROQ: GroqProviderBuilder,
        ProviderType.MISTRAL: MistralProviderBuilder,
        ProviderType.COHERE: CohereProviderBuilder,
        ProviderType.BEDROCK: BedrockProviderBuilder,
    }

    @classmethod
    def get_builder(cls, provider_type: ProviderType) -> ProviderBuilder:
        """
        Get the appropriate builder for the given provider type.
        Args:
            provider_type: The type of provider
        Returns:
            Builder instance for the provider type
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type not in cls._builders:
            raise ValueError(f"Unsupported provider type: {provider_type}")

        return cls._builders[provider_type]()

    @classmethod
    def register_builder(cls, provider_type: ProviderType, builder_class: type[ProviderBuilder]) -> None:
        """
        Register a new provider builder.
        Args:
            provider_type: The provider type to register
            builder_class: The builder class to register
        This allows for extending the factory with custom provider builders.
        """
        cls._builders[provider_type] = builder_class

    @classmethod
    def create_model(
        cls,
        provider: LLMProvider,
        model: LLMModel,
    ) -> Any:
        """
        Create a pydantic-ai model instance from provider and model configuration.
        Args:
            provider: The LLM provider instance
            model: The LLM model instance
        Returns:
            Configured pydantic-ai model instance
        Raises:
            ValueError: If provider type is not supported
        """
        builder = cls.get_builder(provider.type)
        return builder.build_model(provider=provider, model=model)

    @classmethod
    def create_agent(
        cls,
        provider: LLMProvider,
        model: LLMModel,
        toolsets: list,
        system_prompt: str | None = None,
    ) -> Agent:
        """
        Create a pydantic-ai agent from provider and model configuration.
        Args:
            provider: The LLM provider instance
            model: The LLM model instance
            system_prompt: Optional system prompt for the agent
        Returns:
            Configured pydantic-ai Agent instance
        Raises:
            ValueError: If provider type is not supported
        """
        builder = cls.get_builder(provider.type)
        return builder.build_agent(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            toolsets=toolsets,
        )
