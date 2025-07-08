"""Cohere provider builder."""

from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider

from .base import ProviderBuilder


class CohereProviderBuilder(ProviderBuilder):
    """Builder for Cohere providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> CohereModel:
        """
        Build Cohere model with custom provider configuration.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured Cohere model instance
        """
        provider_config = {}

        if provider.api_key:
            provider_config["api_key"] = provider.api_key

        if provider.base_url:
            provider_config["base_url"] = provider.base_url

        # Create provider if we have custom configuration
        if provider_config:
            cohere_provider = CohereProvider(**provider_config)
            return CohereModel(model.name, provider=cohere_provider)
        else:
            return CohereModel(model.name)
