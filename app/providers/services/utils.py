from app.providers.base.token_counter import TokenCounterBase
from app.providers.constants import ProviderType
from app.providers.models import LLMModel, LLMProvider
from app.providers.services.anthropic.token_counter import AnthropicTokenCounter


def get_token_counter(provider: LLMProvider, model: LLMModel) -> TokenCounterBase:
    """
    Get appropriate token counter implementation for provider
    """
    if provider.name == ProviderType.ANTHROPIC:
        return AnthropicTokenCounter(provider=provider, model=model.name)

    raise ValueError(f"Unsupported provider for token counting: {provider.name}")
