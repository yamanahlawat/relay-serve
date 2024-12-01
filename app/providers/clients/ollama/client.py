from datetime import datetime
from typing import AsyncGenerator, Sequence

from loguru import logger
from ollama import AsyncClient, Message, ResponseError

from app.chat.constants import MessageRole, llm_defaults
from app.chat.models import ChatMessage
from app.providers.clients.base import LLMProviderBase
from app.providers.constants import ProviderType
from app.providers.exceptions import (
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from app.providers.factory import ProviderFactory
from app.providers.models import LLMProvider


class OllamaProvider(LLMProviderBase):
    """
    Ollama provider implementation using the official Python SDK.
    """

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self._client = AsyncClient(host=self.provider.base_url)
        self._last_usage = None
        self._available_models = None

    def _get_usage_metrics(self, response: dict) -> dict:
        """
        Extract usage metrics from Ollama response.
        Args:
            response: Raw response from Ollama API
        Returns:
            Dict containing usage metrics
        """
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_duration": response.get("total_duration", 0),  # in nanoseconds
            "load_duration": response.get("load_duration", 0),  # in nanoseconds
            "eval_duration": response.get("eval_duration", 0),  # in nanoseconds
            "created_at": datetime.fromisoformat(response["created_at"].replace("Z", "+00:00")),
            "done_reason": response.get("done_reason"),
        }

    def _prepare_messages(self, messages: Sequence[ChatMessage], new_prompt: str, system_context: str) -> list[Message]:
        """
        Prepare message history for Ollama API.
        Args:
            messages: Previous messages in the conversation
            new_prompt: New user prompt to append
        Returns:
            List of formatted messages for the Ollama API
        """
        formatted_messages = [Message(role="system", content=system_context)]
        for message in messages:
            formatted_messages.append(
                Message(role="assistant" if message.role == MessageRole.ASSISTANT else "user", content=message.content)
            )

        # Add the new user prompt
        formatted_messages.append(Message(role="user", content=new_prompt))
        return formatted_messages

    async def _handle_api_error(self, error: Exception) -> None:
        """
        Handle Ollama API errors and raise appropriate provider exceptions.
        """
        if isinstance(error, ResponseError):
            if error.status_code == 404:
                logger.exception("Ollama model not found")
                raise ProviderConfigurationError(
                    provider=self.provider_type,
                    message="Model not found or not loaded",
                    error=str(error),
                )
            elif error.status_code == 429:
                logger.exception("Ollama API rate limit exceeded")
                raise ProviderRateLimitError(
                    provider=self.provider_type,
                    error=str(error),
                )
            elif error.status_code >= 500:
                logger.exception("Ollama server error")
                raise ProviderConnectionError(
                    provider=self.provider_type,
                    message="Ollama server error",
                    error=str(error),
                )
            else:
                logger.exception("Ollama API error")
                raise ProviderAPIError(
                    provider=self.provider_type,
                    status_code=error.status_code,
                    error=error.error,
                )
        elif isinstance(error, ConnectionError):
            logger.exception("Ollama connection error")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error=str(error),
            )
        elif isinstance(error, TimeoutError):
            logger.exception("Ollama request timed out")
            raise ProviderConnectionError(
                provider=self.provider_type,
                message="Request timed out",
                error=str(error),
            )
        else:
            logger.exception("Unexpected Ollama error")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=500,
                error=str(error),
            )

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        system_context: str = "",
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = llm_defaults.MAX_TOKENS,
        temperature: float = llm_defaults.TEMPERATURE,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate streaming text using Ollama.
        Args:
            prompt: Input prompt text
            model: Model name to use
            messages: Optional previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        Yields:
            Tuple of (chunk text, is_final)
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [], new_prompt=prompt, system_context=system_context
        )
        final_chunk = None

        try:
            total_generated = ""
            stream = self._client.chat(
                model=model,
                messages=formatted_messages,
                stream=True,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )
            async for chunk in await stream:
                # Store final chunk for metrics
                if chunk.get("done", False):
                    final_chunk = chunk

                if content := chunk["message"].get("content", ""):
                    total_generated += content
                    yield (content, False)

            if final_chunk:
                # Get usage metrics from final chunk
                metrics = self._get_usage_metrics(response=final_chunk)
                self._last_usage = (metrics["prompt_tokens"], metrics["completion_tokens"])

            yield ("", True)

        except Exception as error:
            await self._handle_api_error(error)

    async def generate(
        self,
        prompt: str,
        model: str,
        system_context: str = "",
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = llm_defaults.MAX_TOKENS,
        temperature: float = llm_defaults.TEMPERATURE,
    ) -> tuple[str, int, int] | None:
        """
        Generate text using Ollama.
        Args:
            prompt: Input prompt text
            model: Model name to use
            messages: Optional previous conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        Returns:
            Tuple of (generated text, input tokens, output tokens)
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [], new_prompt=prompt, system_context=system_context
        )

        try:
            response = await self._client.chat(
                model=model,
                messages=formatted_messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )

            generated_text = response["message"]["content"]
            metrics = self._get_usage_metrics(response=response)

            # Store usage metrics
            self._last_usage = (metrics["prompt_tokens"], metrics["completion_tokens"])

            return (
                generated_text,
                metrics["prompt_tokens"],
                metrics["completion_tokens"],
            )

        except Exception as error:
            await self._handle_api_error(error)

    def get_token_usage(self) -> tuple[int, int]:
        """
        Get the token usage from the last operation.
        Returns (prompt_tokens, completion_tokens)
        """
        return self._last_usage if self._last_usage else (0, 0)

    async def _fetch_available_models(self) -> list[str]:
        """
        Fetch available models from Ollama API.
        Returns:
            List of model names available in the Ollama instance
        """
        response = await self._client.list()
        return [model["name"] for model in response["models"]]

    @classmethod
    def get_default_models(cls) -> list[str]:
        """
        Get list of default supported models.
        """
        return cls._fetch_available_models()


# Register the Ollama provider with the factory
ProviderFactory.register(provider_type=ProviderType.OLLAMA, provider_class=OllamaProvider)
