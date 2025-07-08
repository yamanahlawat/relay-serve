from typing import Any, AsyncIterator
from uuid import UUID

from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers.factory import ProviderFactory
from app.chat.constants import MessageRole, MessageStatus
from app.chat.schemas.message import MessageCreate, MessageUpdate
from app.chat.services.message import ChatMessageService
from app.chat.services.session import ChatSessionService
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class ChatService:
    """Service for handling chat completions with pydantic_ai"""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the chat service with database session."""
        self.db = db
        self._agents: dict[str, Agent] = {}
        self.message_service = ChatMessageService(db=db)
        self.session_service = ChatSessionService(db=db)

    def _get_agent_key(self, provider: LLMProvider, model: LLMModel) -> str:
        """Generate a unique key for caching agents."""
        return f"{provider.provider_type.value}:{model.name}"

    def _get_or_create_agent(
        self,
        provider: LLMProvider,
        model: LLMModel,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Get or create a cached agent for the provider."""
        cache_key = self._get_agent_key(provider, model)

        if cache_key not in self._agents:
            self._agents[cache_key] = ProviderFactory.create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

        return self._agents[cache_key]

    async def _prepare_conversation_context(
        self,
        session_id: UUID,
        message: str,
    ) -> str:
        """
        Prepare conversation context with recent messages from the session.
        """
        context_parts = []

        # Get recent conversation messages for context
        try:
            recent_messages = await self.message_service.list_messages(
                session_id=session_id,
                offset=0,
                limit=10,  # Get last 10 messages for context
            )

            # Add recent messages to context if found
            if recent_messages:
                message_context = []
                for msg in reversed(recent_messages):  # Reverse to get chronological order
                    role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    content = msg.content or ""
                    message_context.append(f"{role.capitalize()}: {content}")

                if message_context:
                    context_parts.append("Recent conversation:\n" + "\n".join(message_context))
        except SQLAlchemyError as e:
            logger.warning(f"Database error retrieving conversation context: {e}")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Data formatting error in conversation context: {e}")

        # Add the current user message
        context_parts.append(f"User: {message}")

        return "\n\n".join(context_parts)

    async def stream_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        session_id: UUID,
        message_id: UUID,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response for an existing message using pydantic_ai.

        Args:
            provider: LLM provider to use
            model: LLM model to use
            session_id: UUID of the chat session
            message_id: UUID of existing message to complete
            system_prompt: Optional system prompt override
            tools: Optional tools for the agent
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            Streamed response chunks

        Raises:
            ValueError: If message not found or invalid
            RuntimeError: If database or AI operation fails
        """
        try:
            # Get the existing message
            existing_message = await self.message_service.get_message(
                session_id=session_id,
                message_id=message_id,
            )
            if not existing_message or not existing_message.content:
                raise ValueError(f"Message {message_id} not found or has no content")

            message_content = existing_message.content

            # Update message status to processing

            await self.message_service.update_message(
                session_id=session_id,
                message_id=message_id,
                message_in=MessageUpdate(status=MessageStatus.PROCESSING),
            )

            # Get or create agent
            agent = self._get_or_create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Prepare conversation context
            conversation_context = await self._prepare_conversation_context(
                session_id=session_id,
                message=message_content,
            )

            # Prepare model settings
            model_settings_dict = {}
            if temperature is not None:
                model_settings_dict["temperature"] = temperature
            elif model.default_temperature is not None:
                model_settings_dict["temperature"] = model.default_temperature

            if max_tokens is not None:
                model_settings_dict["max_tokens"] = max_tokens
            elif model.default_max_tokens is not None:
                model_settings_dict["max_tokens"] = model.default_max_tokens

            # Track the full response for database storage
            full_response = ""
            model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else None

            # Use pydantic_ai's native streaming
            async with agent.run_stream(conversation_context, model_settings=model_settings) as result:
                async for chunk in result.stream():
                    yield chunk
                    full_response += chunk

            # Add AI response to database after streaming
            if full_response.strip():
                assistant_message = MessageCreate(
                    content=full_response.strip(),
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    parent_id=message_id,
                )
                await self.message_service.create_message(
                    message_in=assistant_message,
                    session_id=session_id,
                )

            # Update original message status to completed
            await self.message_service.update_message(
                session_id=session_id,
                message_id=message_id,
                message_in=MessageUpdate(status=MessageStatus.COMPLETED),
            )

        except ValidationError as e:
            logger.error(f"Validation error in stream_response: {e}", exc_info=True)
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            error_msg = f"\n\n❌ **Error:** Invalid input data.\n\n*Details: {str(e)}*"
            yield error_msg
            raise ValueError(f"Invalid input data: {e}") from e
        except SQLAlchemyError as e:
            logger.error(f"Database error in stream_response: {e}", exc_info=True)
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            error_msg = f"\n\n❌ **Error:** Database operation failed.\n\n*Details: {str(e)}*"
            yield error_msg
            raise RuntimeError(f"Database operation failed: {e}") from e
        except ValueError as e:
            logger.error(f"Value error in stream_response: {e}", exc_info=True)
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            error_msg = f"\n\n❌ **Error:** {str(e)}"
            yield error_msg
            raise
        except Exception as e:
            logger.error(f"Unexpected error streaming response: {e}", exc_info=True)
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            error_msg = f"\n\n❌ **Error:** Unable to generate response. Please try again.\n\n*Details: {str(e)}*"
            yield error_msg
            raise RuntimeError(f"Failed to stream response: {e}") from e


def create_chat_service(db: AsyncSession) -> ChatService:
    """
    Factory function to create a ChatService instance with a database session.
    Args:
        db: Database session
    Returns:
        ChatService instance
    """
    return ChatService(db=db)
