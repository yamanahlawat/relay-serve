from app.core.constants import BaseEnum


class ProviderType(BaseEnum):
    """
    Supported LLM providers
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ClaudeModelName(BaseEnum):
    """
    Anthropic Claude Models
    """

    # Sonnet
    # v3.5 Model
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-latest"

    # Haiku
    # v3.5 Model
    CLAUDE_3_5_HAIKU_LATEST = "claude-3-5-haiku-latest"

    # Opus
    # v3 Model
    CLAUDE_3_OPUS_LATEST = "claude-3-opus-latest"

    # Sonnet
    # v3 Model
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"

    # Haiku
    # v3 Model
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    @classmethod
    def default_models(cls) -> list[str]:
        """
        Get default enabled models
        """
        return [cls.CLAUDE_3_5_SONNET_LATEST.value, cls.CLAUDE_3_5_HAIKU_LATEST.value]


class OpenAIModelName(BaseEnum):
    """
    OpenAI GPT Models
    """

    # GPT-4o Models
    GPT_4O_LATEST = "chatgpt-4o-latest"
    GPT_4O = "gpt-4o"

    # GPT-4o Mini Models
    GPT_4O_MINI = "gpt-4o-mini"

    # O1 Models
    GPT_O1 = "o1-preview"
    GPT_O1_MINI = "o1-mini"

    @classmethod
    def default_models(cls) -> list[str]:
        """Get default enabled models"""
        return [cls.GPT_4O.value, cls.GPT_4O_MINI.value]
