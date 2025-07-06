"""Pydantic AI provider factory and management."""

from typing import Any

from pydantic_ai import Agent

from app.core.config import settings
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class PydanticAIProviderFactory:
    """Factory for creating pydantic_ai models and agents."""

    @staticmethod
    def create_model(provider: LLMProvider, model: LLMModel) -> str:
        """Create a pydantic_ai model string from provider and model configuration."""
        # The provider name should directly correspond to what pydantic_ai expects
        provider_name = provider.name.lower()

        # Return the model string that pydantic_ai can use
        return f"{provider_name}:{model.name}"

    @staticmethod
    def create_agent(
        provider: LLMProvider,
        model: LLMModel,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Create a pydantic_ai agent from provider and model configuration."""
        model_string = PydanticAIProviderFactory.create_model(provider, model)

        # Create agent with model string - pydantic_ai will handle provider details
        agent_kwargs = {
            "model": model_string,
            "api_key": provider.api_key,
            "retries": settings.LLM.DEFAULT_RETRIES,
        }

        if system_prompt:
            agent_kwargs["system_prompt"] = system_prompt

        if tools:
            agent_kwargs["tools"] = tools

        return Agent(**agent_kwargs)
