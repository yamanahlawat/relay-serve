from app.core.constants import BaseEnum


class ProviderType(BaseEnum):
    """
    Supported LLM providers
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ProviderErrorCode(BaseEnum):
    """
    Error codes for provider operations
    """

    CONNECTION = "connection_error"
    RATE_LIMIT = "rate_limit_error"
    API = "api_error"
    CONFIGURATION = "configuration_error"
