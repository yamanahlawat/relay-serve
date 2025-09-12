"""OpenAI provider builder."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.ai.providers.base import ProviderBuilder
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class OpenAIProviderBuilder(ProviderBuilder):
    """Builder for OpenAI and OpenAI-compatible providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> OpenAIChatModel:
        """
        Build OpenAI model with custom provider configuration.

        Supports custom base URLs for OpenAI-compatible APIs like:
        - Local models (Ollama, etc.)
        - AI gateways (OpenRouter, etc.)
        - Custom deployments

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured OpenAI model instance
        """
        provider_config = {}

        if provider.api_key:
            provider_config["api_key"] = provider.api_key

        if provider.base_url:
            provider_config["base_url"] = provider.base_url

        # Create provider if we have custom configuration
        if provider_config:
            openai_provider = OpenAIProvider(**provider_config)
            return OpenAIChatModel(model_name=model.name, provider=openai_provider)
        else:
            # Use default provider with environment variables
            return OpenAIChatModel(model_name=model.name)
