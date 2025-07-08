"""Groq provider builder."""

from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider

from .base import ProviderBuilder


class GroqProviderBuilder(ProviderBuilder):
    """Builder for Groq providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> GroqModel:
        """
        Build Groq model with custom provider configuration.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured Groq model instance
        """
        provider_config = {}

        if provider.api_key:
            provider_config["api_key"] = provider.api_key

        if provider.base_url:
            provider_config["base_url"] = provider.base_url

        # Create provider if we have custom configuration
        if provider_config:
            groq_provider = GroqProvider(**provider_config)
            return GroqModel(model.name, provider=groq_provider)
        else:
            return GroqModel(model.name)
