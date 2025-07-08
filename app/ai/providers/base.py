"""Base provider builder interface."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic_ai import Agent

from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class ProviderBuilder(ABC):
    """Abstract base class for provider-specific model and agent builders."""

    @abstractmethod
    def build_model(self, provider: LLMProvider, model: LLMModel) -> Any:
        """
        Build a pydantic-ai model instance for the given provider and model.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance

        Returns:
            Configured pydantic-ai model instance
        """
        pass

    def build_agent(
        self,
        provider: LLMProvider,
        model: LLMModel,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """
        Build a pydantic-ai agent for the given provider and model.

        This is a concrete implementation that works for all providers
        by delegating model creation to the provider-specific build_model method.

        Args:
            provider: The LLM provider instance
            model: The LLM model instance
            system_prompt: Optional system prompt for the agent
            tools: Optional list of tools for the agent

        Returns:
            Configured pydantic-ai Agent instance
        """
        pydantic_model = self.build_model(provider, model)

        agent_kwargs = {
            "model": pydantic_model,
            "tools": tools or [],
        }

        if system_prompt is not None:
            agent_kwargs["system_prompt"] = system_prompt

        return Agent(**agent_kwargs)
