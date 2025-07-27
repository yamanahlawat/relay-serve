"""AWS Bedrock provider builder."""

from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from app.ai.providers.base import ProviderBuilder
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class BedrockProviderBuilder(ProviderBuilder):
    """Builder for AWS Bedrock providers."""

    def build_model(self, provider: LLMProvider, model: LLMModel) -> BedrockConverseModel:
        """
        Build Bedrock model with custom provider configuration.

        Note: Bedrock typically uses AWS credentials (access keys, IAM roles, etc.)
        rather than simple API keys. Configuration is usually handled through
        AWS SDK configuration (environment variables, profiles, etc.).

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured Bedrock model instance
        """
        provider_config = {}

        # Bedrock typically uses AWS credentials rather than API keys
        # Custom configuration would be handled through AWS SDK configuration
        if provider.base_url:
            # This might not be applicable for Bedrock, but we'll include it for completeness
            provider_config["base_url"] = provider.base_url

        # Create provider if we have custom configuration
        if provider_config:
            bedrock_provider = BedrockProvider(**provider_config)
            return BedrockConverseModel(model_name=model.name, provider=bedrock_provider)
        else:
            return BedrockConverseModel(model_name=model.name)
