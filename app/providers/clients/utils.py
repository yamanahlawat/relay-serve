from app.providers.clients.anthropic.token_counter import AnthropicTokenCounter
from app.providers.clients.base import TokenCounterBase
from app.providers.clients.openai.token_counter import OpenAITokenCounter
from app.providers.constants import ProviderType
from app.providers.models import LLMModel, LLMProvider


def get_token_counter(provider: LLMProvider, model: LLMModel) -> TokenCounterBase | None:
    """
    Get appropriate token counter implementation for provider
    """
    # Skip token counting if model has no costs
    if model.input_cost_per_token == 0 and model.output_cost_per_token == 0:
        return None

    if provider.type == ProviderType.ANTHROPIC:
        return AnthropicTokenCounter(provider=provider, model=model.name)

    elif provider.type == ProviderType.OPENAI:
        return OpenAITokenCounter(provider=provider, model=model.name)

    elif provider.type == ProviderType.OLLAMA:
        # Ollama doesn't need token counting
        return None

    raise ValueError(f"Unsupported provider for token counting: {provider.type}")
