"""Anthropic provider builder."""

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from app.ai.providers.base import ProviderBuilder
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class AnthropicProviderBuilder(ProviderBuilder):
    """Builder for Anthropic Claude providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> AnthropicModel:
        """
        Build Anthropic model with custom provider configuration.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured Anthropic model instance
        """
        provider_config = {}

        if provider.api_key:
            provider_config["api_key"] = provider.api_key

        if provider.base_url:
            provider_config["base_url"] = provider.base_url

        # Create provider if we have custom configuration
        if provider_config:
            anthropic_provider = AnthropicProvider(**provider_config)
            return AnthropicModel(model_name=model.name, provider=anthropic_provider)
        else:
            return AnthropicModel(model_name=model.name)
