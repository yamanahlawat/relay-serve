from typing import AsyncGenerator, Sequence
from uuid import UUID

from langfuse.decorators import langfuse_context, observe
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.constants import MessageRole, MessageStatus
from app.chat.crud import crud_message
from app.chat.models import ChatMessage, ChatSession
from app.chat.schemas import MessageCreate
from app.chat.schemas.chat import CompletionParams, CompletionRequest, CompletionResponse
from app.chat.schemas.message import MessageUsage
from app.chat.services import ChatSessionService
from app.chat.services.message import ChatMessageService
from app.providers.clients import AnthropicProvider, OllamaProvider, OpenAIProvider
from app.providers.clients.base import LLMProviderBase
from app.providers.clients.utils import get_token_counter
from app.providers.factory import ProviderFactory
from app.providers.models import LLMModel, LLMProvider


class ChatCompletionService:
    """
    Service for handling chat completions
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.message_service = ChatMessageService(db=self.db)

    async def validate_request(
        self,
        session_id: UUID,
    ) -> tuple[ChatSession, LLMProvider, LLMModel]:
        """
        Validate the chat completion request and return required models.
        Args:
            session_id: UUID of the chat session
            request: Chat completion request
        Returns:
            Tuple of (ChatSession, LLMProvider, LLMModel)
        Raises:
            HTTPException: If session, provider or model not found
        """
        # Verify session exists and is active
        session_service = ChatSessionService(db=self.db)
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
        Raises:
            HTTPException: If session, message not found
        """
        active_session, provider, llm_model = await self.validate_request(session_id=session_id)
        # Get message and verify it belongs to session
        await self.message_service.get_message(session_id=session_id, message_id=message_id)
        return active_session, provider, llm_model

    async def create_user_message(
        self,
        session_id: UUID,
        content: str,
        provider: LLMProvider,
        model: LLMModel,
        parent_id: UUID | None = None,
    ) -> ChatMessage:
        """
        Create a new user message with token counting and cost calculation.
        """

        # Get token counter for provider/model
        token_counter = get_token_counter(provider=provider, model=model)

        # Count input tokens
        input_tokens = await token_counter.count_tokens(content) if token_counter else 0

        # Calculate cost
        input_cost = input_tokens * model.input_cost_per_token

        return await crud_message.create_with_session(
            db=self.db,
            session_id=session_id,
            obj_in=MessageCreate(
                content=content,
                role=MessageRole.USER,
                status=MessageStatus.COMPLETED,
                parent_id=parent_id,
                usage=MessageUsage(
                    input_tokens=input_tokens,
                    output_tokens=0,
                    input_cost=input_cost,
                    output_cost=0,
                ),
            ),
        )

    def get_provider_client(self, provider: LLMProvider) -> LLMProviderBase:
        """
        Get the appropriate provider client.
        Args:
            provider: LLM provider configuration
        Returns:
            Provider-specific client instance
        Raises:
            HTTPException: If provider type is not supported
        """
        return ProviderFactory.get_client(provider=provider)

    async def get_conversation_history(
        self,
        chat_session: ChatSession,
        current_message_id: UUID | None = None,
    ) -> Sequence[ChatMessage]:
        """
        Get conversation history for context.
        """
        return await crud_message.get_session_context(
            db=self.db,
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

        # Create message with usage metrics
        assistant_message = await crud_message.create_with_session(
            db=self.db,
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

    def update_langfuse_trace(
        self,
        chat_session: ChatSession,
        assistant_message: ChatMessage,
        user_message: ChatMessage,
        model: str,
        usage: dict,
        model_parameters: dict,
    ) -> None:
        """
        Update langfuse trace with complete context.
        """
        langfuse_context.update_current_observation(
            session_id=str(chat_session.id),
            status_message=assistant_message.status,
            input=user_message.content,
            output=assistant_message.content,
            model=model,
            usage=usage,
            model_parameters=model_parameters,
            metadata={
                "model_id": chat_session.llm_model_id,
                "provider_id": chat_session.provider_id,
                "message_id": assistant_message.id,
            },
        )

    @observe(name="streaming")
    async def generate_stream(
        self,
        chat_session: ChatSession,
        model: LLMModel,
        provider_client: AnthropicProvider | OllamaProvider | OpenAIProvider,
        params: CompletionParams,
        message_id: UUID,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        # Get the message
        message = await self.message_service.get_message(
            session_id=chat_session.id,
            message_id=message_id,
        )

        history = await self.get_conversation_history(
            chat_session=chat_session,
            current_message_id=message_id,
        )
        full_content = ""

        async for chunk, is_final in provider_client.generate_stream(
            prompt=message.content,
            model=model.name,
            system_context=chat_session.system_context,
            messages=history,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
        ):
            if chunk:
                full_content += chunk
                yield chunk

            if is_final:
                input_tokens, output_tokens = provider_client.get_token_usage()
                assistant_message = await self.create_assistant_message(
                    chat_session=chat_session,
                    content=full_content,
                    parent_id=message_id,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

                self.update_langfuse_trace(
                    chat_session=chat_session,
                    assistant_message=assistant_message,
                    user_message=message,
                    model=model.name,
                    usage={
                        "input": input_tokens,
                        "output": output_tokens,
                    },
                    model_parameters={
                        "max_tokens": params.max_tokens,
                        "temperature": params.temperature,
                    },
                )

    @observe(name="non_streaming")
    async def generate_complete(
        self,
        chat_session: ChatSession,
        model: LLMModel,
        provider_client: AnthropicProvider | OllamaProvider | OpenAIProvider,
        request: CompletionRequest,
        user_message: ChatMessage,
    ) -> CompletionResponse:
        """
        Generate complete response
        """
        # Get conversation history
        history = await self.get_conversation_history(
            chat_session=chat_session,
            current_message_id=user_message.id,
        )

        # Generate response with history
        content, input_tokens, output_tokens = await provider_client.generate(
            prompt=request.prompt,
            model=model.name,
            system_context=chat_session.system_context,
            messages=history,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        assistant_message = await self.create_assistant_message(
            chat_session=chat_session,
            content=content,
            parent_id=user_message.id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Update Langfuse context with the assistant message
        self.update_langfuse_trace(
            chat_session=chat_session,
            assistant_message=assistant_message,
            user_message=user_message,
            model=model.name,
            usage={"input": input_tokens, "output": output_tokens},
            model_parameters={
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        return CompletionResponse(
            content=content,
            model=model.name,
            provider=provider_client.provider.name,
            usage=assistant_message.get_usage(),
        )
