"""Gemini provider builder."""

from pydantic_ai.models.gemini import GeminiModel

from app.ai.providers.base import ProviderBuilder
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class GeminiProviderBuilder(ProviderBuilder):
    """Builder for Google Gemini providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> GeminiModel:
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
        return GeminiModel(model_name=model.name)
