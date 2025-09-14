"""
Provider types for LLM providers based on pydantic_ai support.
"""

from app.core.constants import BaseEnum


class ProviderType(BaseEnum):
    """
    Supported LLM provider types based on pydantic_ai documentation.

    This enum defines the provider types that users can select from when
    configuring their LLM providers in the UI.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    BEDROCK = "bedrock"
