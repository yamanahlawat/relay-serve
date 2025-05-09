from typing import Any, AsyncGenerator, Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.constants import MessageRole, MessageStatus, error_messages
from app.chat.crud import crud_message
from app.chat.models import ChatMessage, ChatSession
from app.chat.schemas import MessageCreate
from app.chat.schemas.chat import CompletionParams
from app.chat.schemas.message import MessageRead, MessageUsage
from app.chat.services import ChatSessionService
from app.chat.services.message import ChatMessageService
from app.database.session import AsyncSessionLocal
from app.model_context_protocol.services.tool_execution import mcp_tool_service
from app.providers.clients import AnthropicProvider, OllamaProvider, OpenAIProvider
from app.providers.clients.base import LLMProviderBase
from app.providers.exceptions.client import ProviderAPIError, ProviderConnectionError, ProviderRateLimitError
from app.providers.factory import ProviderFactory
from app.providers.models import LLMModel, LLMProvider


class ChatCompletionService:
    """
    Service for handling chat completions.
    """

    def __init__(
        self,
        provider_factory: ProviderFactory,
    ) -> None:
        self.provider_factory = provider_factory
        self.mcp_service = mcp_tool_service

    async def validate_request(
        self,
        db: AsyncSession,
        session_id: UUID,
    ) -> tuple[ChatSession, LLMProvider, LLMModel]:
        """
        Validate the chat completion request and return required models.
        Args:
            session_id: UUID of the chat session
        Returns:
            Tuple of (ChatSession, LLMProvider, LLMModel)
        """
        # Verify session exists and is active
        session_service = ChatSessionService(db=db)
        active_session = await session_service.get_active_session(session_id=session_id)
        return active_session, active_session.provider, active_session.llm_model

    async def validate_message(
        self,
        session_id: UUID,
        message_id: UUID,
    ) -> tuple[ChatSession, LLMProvider, LLMModel]:
        """
        Validate the message and return required models.
        Args:
            session_id: UUID of the chat session
            message_id: UUID of the message
        Returns:
            Tuple of (ChatSession, LLMProvider, LLMModel)
        """
        async with AsyncSessionLocal() as db:
            active_session, provider, llm_model = await self.validate_request(db=db, session_id=session_id)
            # Get message and verify it belongs to session
            message_service = ChatMessageService(db=db)
            await message_service.get_message(session_id=session_id, message_id=message_id)
            return active_session, provider, llm_model

    async def create_user_message(
        self,
        session_id: UUID,
        content: str,
        parent_id: UUID | None = None,
    ) -> ChatMessage:
        """
        Create a new user message.
        """
        async with AsyncSessionLocal() as db:
            return await crud_message.create(
                db=db,
                session_id=session_id,
                obj_in=MessageCreate(
                    content=content,
                    role=MessageRole.USER,
                    status=MessageStatus.COMPLETED,
                    parent_id=parent_id,
                ),
            )

    def get_provider_client(self, provider: LLMProvider) -> LLMProviderBase:
        """
        Get the appropriate provider client.
        Args:
            provider: LLM provider configuration
        Returns:
            Provider-specific client instance
        """
        return self.provider_factory.get_client(provider=provider)

    async def get_conversation_history(
        self,
        chat_session: ChatSession,
        current_message_id: UUID | None = None,
    ) -> Sequence[ChatMessage]:
        """
        Get conversation history for context.
        """
        async with AsyncSessionLocal() as db:
            return await crud_message.get_session_context(
                db=db,
                session_id=chat_session.id,
                exclude_message_id=current_message_id,
            )

    async def create_assistant_message(
        self,
        chat_session: ChatSession,
        content: str,
        parent_id: UUID,
        model: LLMModel,
        input_tokens: int,
        output_tokens: int,
    ) -> ChatMessage:
        """
        Create an assistant message and update usage tracking.
        """
        # Calculate costs using model's rates
        input_cost = input_tokens * model.input_cost_per_token
        output_cost = output_tokens * model.output_cost_per_token

        async with AsyncSessionLocal() as db:
            # Create message with usage metrics
            assistant_message = await crud_message.create(
                db=db,
                session_id=chat_session.id,
                obj_in=MessageCreate(
                    content=content,
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    parent_id=parent_id,
                    usage=MessageUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        input_cost=input_cost,
                        output_cost=output_cost,
                    ),
                ),
            )
            return assistant_message

    def get_model_params(
        self,
        model: LLMModel,
        request_params: CompletionParams,
    ) -> dict:
        """
        Get model parameters with request overrides.
        Precedence: Request params > Model params > Global defaults
        """
        # Request params override model params
        return {
            "max_tokens": request_params.max_tokens or model.max_tokens,
            "temperature": request_params.temperature or model.temperature,
            "top_p": request_params.top_p or model.top_p,
        }

    async def handle_provider_error(self, error: Exception) -> str:
        """
        Handle provider errors and return user-friendly messages.
        """
        if isinstance(error, ProviderRateLimitError):
            return error_messages.RATE_LIMIT_ERROR
        elif isinstance(error, (ProviderConnectionError, ProviderAPIError)):
            return error_messages.PROVIDER_ERROR
        return error_messages.GENERAL_ERROR

    async def finalize_assistant_message(
        self,
        chat_session: ChatSession,
        message_id: UUID,
        model: LLMModel,
        current_message: ChatMessage,
        full_content: str,
        params: CompletionParams,
        provider_client: LLMProviderBase,
        extra_data: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Finalize the assistant message with complete content and metadata.
        """

        input_tokens, output_tokens = provider_client.get_token_usage()

        async with AsyncSessionLocal() as db:
            # Create message with usage metrics and metadata
            return await crud_message.create(
                db=db,
                session_id=chat_session.id,
                obj_in=MessageCreate(
                    content=full_content,
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    parent_id=message_id,
                    usage=MessageUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        input_cost=input_tokens * model.input_cost_per_token,
                        output_cost=output_tokens * model.output_cost_per_token,
                    ),
                    extra_data=extra_data or {},
                ),
            )

    async def generate_chat_stream(
        self,
        chat_session: ChatSession,
        model: LLMModel,
        provider_client: AnthropicProvider | OllamaProvider | OpenAIProvider,
        params: CompletionParams,
        message_id: UUID,
    ) -> AsyncGenerator[dict[str, Any] | Any | str, Any]:
        """
        Generate streaming response.
        Ensures that if the stream ends without a final flag, the assistant message is still created.
        Handles:
        - Regular content streaming
        - Tool execution and results
        - Model thinking states
        - Errors and completion states

        Yields a formatted string representation of each state for the frontend.
        """
        async with AsyncSessionLocal() as db:
            message_service = ChatMessageService(db=db)
            current_message = await message_service.get_message(
                session_id=chat_session.id,
                message_id=message_id,
            )

        history = await self.get_conversation_history(
            chat_session=chat_session,
            current_message_id=message_id,
        )

        available_tools = await self.mcp_service.get_available_tools()

        async for block, metadata in provider_client.generate_stream(
            current_message=current_message,
            model=model.name,
            system_context=chat_session.system_context,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            messages=history,
            session_id=chat_session.id,
            available_tools=available_tools,
        ):
            # Send block to client
            if block.type == "done":
                # Store metadata if received
                if metadata:
                    # Create final message with metadata
                    final_message = await self.finalize_assistant_message(
                        chat_session=chat_session,
                        message_id=message_id,
                        model=model,
                        current_message=current_message,
                        full_content=metadata.content,
                        params=params,
                        provider_client=provider_client,
                        extra_data={
                            "stream_blocks": [tool.model_dump() for tool in metadata.stream_blocks],
                        },
                    )
                    block.message = MessageRead(
                        id=final_message.id,
                        session_id=final_message.session_id,
                        role=MessageRole.ASSISTANT,
                        content=metadata.content,
                        status=MessageStatus.COMPLETED,
                        parent_id=message_id,
                        created_at=final_message.created_at,
                        usage=final_message.get_usage(),
                        attachments=final_message.attachments,
                        error_code=None,
                        error_message=None,
                        extra_data={"stream_blocks": metadata.stream_blocks},
                    )

            yield block.model_dump_json(exclude_unset=True)
