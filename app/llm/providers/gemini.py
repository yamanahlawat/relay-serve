"""Gemini provider builder."""

from pydantic_ai.models.google import GoogleModel

from app.llm.providers.base import ProviderBuilder
from app.model.model import LLMModel
from app.provider.model import LLMProvider


class GeminiProviderBuilder(ProviderBuilder):
    """Builder for Google Gemini providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> GoogleModel:
        """
        Build Gemini model with custom provider configuration.

        Note: Gemini typically uses Google Cloud authentication or API keys
        configured through environment variables. Custom base URLs are not
        commonly supported.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured Gemini model instance
        """
        # For Gemini, we typically use the default provider
        # Custom configuration would need to be handled through environment variables
        # or custom client configuration
        return GoogleModel(model_name=model.name)
