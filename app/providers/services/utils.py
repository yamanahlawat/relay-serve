from app.providers.base.token_counter import TokenCounterBase
from app.providers.constants import ProviderType
from app.providers.models import LLMModel, LLMProvider
from app.providers.services.anthropic.token_counter import AnthropicTokenCounter
from app.providers.services.openai.token_counter import OpenAITokenCounter


def get_token_counter(provider: LLMProvider, model: LLMModel) -> TokenCounterBase | None:
    """
    Get appropriate token counter implementation for provider
    """
    if provider.name == ProviderType.ANTHROPIC:
        return AnthropicTokenCounter(provider=provider, model=model.name)

    elif provider.name == ProviderType.OPENAI:
        return OpenAITokenCounter(provider=provider, model=model.name)

    elif provider.name == ProviderType.OLLAMA:
        # Ollama doesn't need token counting
        return None

    raise ValueError(f"Unsupported provider for token counting: {provider.name}")
