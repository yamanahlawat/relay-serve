from typing import AsyncGenerator, Sequence
from uuid import UUID

from fastapi import HTTPException, status
from langfuse.decorators import langfuse_context, observe
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.constants import MessageRole, MessageStatus
from app.chat.crud import crud_message, crud_session
from app.chat.models import ChatMessage, ChatSession
from app.chat.schemas import ChatRequest, ChatResponse, MessageCreate
from app.chat.services.usage import UsageTracker
from app.providers.constants import ProviderType
from app.providers.crud.model import crud_model
from app.providers.crud.provider import crud_provider
from app.providers.models import LLMModel, LLMProvider
from app.providers.services.anthropic.client import AnthropicProvider
from app.providers.services.utils import get_token_counter


class ChatCompletionService:
    """
    Service for handling chat completions
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.usage_tracker = UsageTracker(db=db)

    async def validate_request(
        self,
        session_id: UUID,
        request: ChatRequest,
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
        chat_session = await crud_session.get_active(db=self.db, id=session_id)
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active chat session {session_id} not found",
            )

        # Get provider
        provider = await crud_provider.get(db=self.db, id=request.provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {request.provider_id} not found",
            )

        # Get model
        model = await crud_model.get(db=self.db, id=request.llm_model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.llm_model_id} not found",
            )

        return chat_session, provider, model

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
        input_tokens = await token_counter.count_tokens(content)

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
                input_tokens=input_tokens,
                output_tokens=0,
                input_cost=input_cost,
                output_cost=0,
            ),
        )

    def get_provider_client(self, provider: LLMProvider) -> AnthropicProvider:
        """
        Get the appropriate provider client.
        Args:
            provider: LLM provider configuration
        Returns:
            Provider-specific client instance
        Raises:
            HTTPException: If provider type is not supported
        """
        if provider.name != ProviderType.ANTHROPIC:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider: {provider.name}",
            )
        return AnthropicProvider(provider=provider)

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
        assistant_message = await crud_message.create_with_session(
            db=self.db,
            session_id=chat_session.id,
            obj_in=MessageCreate(
                content=content,
                role=MessageRole.ASSISTANT,
                status=MessageStatus.COMPLETED,
                parent_id=parent_id,
            ),
        )

        await self.usage_tracker.update_message_usage(
            message_id=assistant_message.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

        return assistant_message

    def update_langfuse_observation(
        self,
        chat_session: ChatSession,
        assistant_message: ChatMessage,
        user_message: ChatMessage,
    ) -> None:
        """
        Update Langfuse observation with message details.
        """

        # Update the observation with output details
        langfuse_context.update_current_observation(
            session_id=str(chat_session.id),
            status_message=assistant_message.status,
            input=user_message.content,
            output=assistant_message.content,
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
        provider_client: AnthropicProvider,
        request: ChatRequest,
        user_message: ChatMessage,
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response
        """
        # Get conversation history
        history = await self.get_conversation_history(
            chat_session=chat_session,
            current_message_id=user_message.id,
        )

        full_content = ""
        input_tokens = output_tokens = 0

        async for chunk, is_final in provider_client.generate_stream(
            prompt=request.prompt,
            model=model.name,
            system_context=chat_session.system_context,
            messages=history,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            if chunk:  # If there's content
                full_content += chunk
                yield chunk

            if is_final:  # Final message with usage info
                # Get all tokens (input + output) from provider's response
                input_tokens, output_tokens = provider_client.get_token_usage()
                assistant_message = await self.create_assistant_message(
                    chat_session=chat_session,
                    content=full_content,
                    parent_id=user_message.id,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

                # Update Langfuse context with the assistant message
                self.update_langfuse_observation(
                    chat_session=chat_session,
                    assistant_message=assistant_message,
                    user_message=user_message,
                )

    # @observe(name="non_streaming")
    async def generate_complete(
        self,
        chat_session: ChatSession,
        model: LLMModel,
        provider_client: AnthropicProvider,
        request: ChatRequest,
        user_message: ChatMessage,
    ) -> ChatResponse:
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
        self.update_langfuse_observation(
            chat_session=chat_session,
            assistant_message=assistant_message,
            user_message=user_message,
        )

        return ChatResponse(
            content=content,
            model=model.name,
            provider=provider_client.provider.name,
            usage=assistant_message.get_usage(),
        )
