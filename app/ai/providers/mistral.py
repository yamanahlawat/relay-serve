"""Mistral provider builder."""

from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider

from .base import ProviderBuilder


class MistralProviderBuilder(ProviderBuilder):
    """Builder for Mistral providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> MistralModel:
        """
        Build Mistral model with custom provider configuration.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured Mistral model instance
        """
        provider_config = {}

        if provider.api_key:
            provider_config["api_key"] = provider.api_key

        if provider.base_url:
            provider_config["base_url"] = provider.base_url

        # Create provider if we have custom configuration
        if provider_config:
            mistral_provider = MistralProvider(**provider_config)
            return MistralModel(model.name, provider=mistral_provider)
        else:
            return MistralModel(model.name)
