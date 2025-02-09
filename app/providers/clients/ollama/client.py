from datetime import datetime
from typing import AsyncGenerator, Sequence

from loguru import logger
from ollama import AsyncClient, Image, Message, ResponseError

from app.chat.constants import AttachmentType, MessageRole
from app.chat.models import ChatMessage
from app.chat.services.sse import get_sse_manager
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
        # Instantiate and cache the AsyncClient for reuse.
        self._client = AsyncClient(host=self.provider.base_url)
        self._last_usage: tuple[int, int] | None = None

    def _get_usage_metrics(self, response: dict) -> dict:
        """
        Extract usage metrics from the Ollama response.
        Args:
            response: Raw response from the Ollama API.
        Returns:
            A dictionary containing usage metrics.
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

    def _prepare_messages(
        self,
        messages: Sequence[ChatMessage],
        current_message: ChatMessage,
        system_context: str,
    ) -> list[Message]:
        """
        Prepare message history for the Ollama API.
        Args:
            messages: Previous messages in the conversation.
            new_prompt: New user prompt to append.
            system_context: System context to send as a system message.
        Returns:
            A list of formatted messages for the Ollama API.
        """
        formatted_messages = [Message(role="system", content=system_context)]
        for message in messages:
            role = "assistant" if message.role == MessageRole.ASSISTANT else "user"
            formatted_messages.append(Message(role=role, content=message.content))
        # Append the new prompt as a user message.
        message_images = []
        for attachment in current_message.direct_attachments:
            if attachment.type == AttachmentType.IMAGE.value:
                message_images.append(Image(value=attachment.storage_path))
        formatted_messages.append(
            Message(
                role="user",
                content=current_message.content,
                images=message_images,
            )
        )
        return formatted_messages

    async def _handle_api_error(self, error: Exception) -> None:
        """
        Handle Ollama API errors and raise the appropriate provider exceptions.
        Uses exception chaining to preserve the original traceback.
        """
        if isinstance(error, ResponseError):
            if error.status_code == 404:
                logger.exception("Ollama model not found")
                raise ProviderConfigurationError(
                    provider=self.provider_type,
                    message="Model not found or not loaded",
                    error=str(error),
                ) from error
            elif error.status_code == 429:
                logger.exception("Ollama API rate limit exceeded")
                raise ProviderRateLimitError(
                    provider=self.provider_type,
                    error=str(error),
                ) from error
            elif error.status_code >= 500:
                logger.exception("Ollama server error")
                raise ProviderConnectionError(
                    provider=self.provider_type,
                    message="Ollama server error",
                    error=str(error),
                ) from error
            else:
                logger.exception("Ollama API error")
                raise ProviderAPIError(
                    provider=self.provider_type,
                    status_code=error.status_code,
                    error=error.error,
                ) from error
        elif isinstance(error, ConnectionError):
            logger.exception("Ollama connection error")
            raise ProviderConnectionError(
                provider=self.provider_type,
                error=str(error),
            ) from error
        elif isinstance(error, TimeoutError):
            logger.exception("Ollama request timed out")
            raise ProviderConnectionError(
                provider=self.provider_type,
                message="Request timed out",
                error=str(error),
            ) from error
        else:
            logger.exception("Unexpected Ollama error")
            raise ProviderAPIError(
                provider=self.provider_type,
                status_code=500,
                error=str(error),
            ) from error

    async def generate_stream(
        self,
        current_message: ChatMessage,
        model: str,
        system_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        messages: Sequence[ChatMessage] | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate streaming text using Ollama.
        Yields:
            A tuple of (text chunk, is_final). An empty string with is_final=True indicates completion.
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [],
            current_message=current_message,
            system_context=system_context,
        )
        cancel_key = f"sse:cancel:{session_id}" if session_id else None
        sse_manager = await get_sse_manager()
        final_chunk = None  # To store the final chunk containing usage metrics.

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            stream = self._client.chat(
                model=model,
                messages=formatted_messages,
                stream=True,
                options=options,
            )
            async for chunk in await stream:
                if cancel_key and await sse_manager.redis.exists(cancel_key):
                    logger.warning(f"Stream cancelled for session {session_id}")
                    stream.close()  # Explicitly close the stream from the SDK
                    break

                # Store the final chunk for usage metrics.
                if chunk.get("done", False):
                    final_chunk = chunk
                if content := chunk.get("message", {}).get("content", ""):
                    yield (content, False)

            if final_chunk is not None:
                # Get usage metrics from final chunk
                metrics = self._get_usage_metrics(response=final_chunk)
                self._last_usage = (metrics["prompt_tokens"], metrics["completion_tokens"])

            yield ("", True)

        except Exception as error:
            await self._handle_api_error(error)

    async def generate(
        self,
        current_message: ChatMessage,
        model: str,
        system_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        messages: Sequence[ChatMessage] | None = None,
    ) -> tuple[str, int, int] | None:
        """
        Generate text using Ollama.
        Returns:
            A tuple containing (generated text, input tokens, output tokens).
        """
        formatted_messages = self._prepare_messages(
            messages=messages or [], current_message=current_message, system_context=system_context
        )
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            response = await self._client.chat(
                model=model,
                messages=formatted_messages,
                options=options,
            )
            generated_text = response.get("message", {}).get("content", "")
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
        Get token usage from the last operation.
        Returns:
            A tuple (prompt_tokens, completion_tokens).
        """
        return self._last_usage if self._last_usage is not None else (0, 0)

    async def list_existing_models(self) -> list[str]:
        """
        Fetch available models from the Ollama API.
        Returns:
            A list of model names available in the Ollama instance.
        """
        response = await self._client.list()
        return [model["name"] for model in response.get("models", [])]


# Register the Ollama provider with the factory.
ProviderFactory.register(provider_type=ProviderType.OLLAMA, provider_class=OllamaProvider)
