from typing import Any, AsyncGenerator, Sequence
from uuid import UUID

from langfuse.decorators import langfuse_context, observe
from mcp.types import EmbeddedResource, ImageContent, TextContent
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.constants import MessageRole, MessageStatus, error_messages
from app.chat.crud import crud_message
from app.chat.models import ChatMessage, ChatSession
from app.chat.schemas import MessageCreate
from app.chat.schemas.chat import CompletionParams, CompletionRequest, CompletionResponse
from app.chat.schemas.common import ChatUsage
from app.chat.schemas.message import MessageUsage
from app.chat.schemas.stream import StreamBlockType
from app.chat.services import ChatSessionService
from app.chat.services.message import ChatMessageService
from app.model_context_protocol.services.tool import MCPService
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
        db: AsyncSession,
        provider_factory: ProviderFactory,
    ) -> None:
        self.db = db
        self.provider_factory = provider_factory
        self.message_service = ChatMessageService(db=self.db)
        self.mcp_service = MCPService()

    async def validate_request(
        self,
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
        Create a new user message.
        """
        return await crud_message.create(
            db=self.db,
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
        assistant_message = await crud_message.create(
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
        Update Langfuse trace with complete context.
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Finalize the assistant message with complete content and metadata.
        """

        input_tokens, output_tokens = provider_client.get_token_usage()

        # Create message with usage metrics and metadata
        assistant_message = await crud_message.create(
            db=self.db,
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
                extra_data=metadata or {},
            ),
        )

        # Update Langfuse trace
        self.update_langfuse_trace(
            chat_session=chat_session,
            assistant_message=assistant_message,
            user_message=current_message,
            model=model.name,
            usage={"input": input_tokens, "output": output_tokens},
            model_parameters={
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
            },
        )

    def _format_tool_content_for_display(self, content: list[TextContent | ImageContent | EmbeddedResource]) -> str:
        """
        Format tool content for display in chat
        """
        parts = []

        for item in content:
            if isinstance(item, TextContent):
                parts.append(item.text)
            else:
                raise NotImplementedError("ImageContent and EmbeddedResource not supported yet")
        return "\n".join(parts)

    @observe(name="streaming")
    async def generate_chat_stream(
        self,
        chat_session: ChatSession,
        model: LLMModel,
        provider_client: AnthropicProvider | OllamaProvider | OpenAIProvider,
        params: CompletionParams,
        message_id: UUID,
    ) -> AsyncGenerator[str, None]:
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
        full_content = []  # List to store all content parts
        tool_outputs = []  # List to store tool results

        try:
            current_message = await self.message_service.get_message(session_id=chat_session.id, message_id=message_id)

            history = await self.get_conversation_history(chat_session=chat_session, current_message_id=message_id)

            available_tools = await self.mcp_service.get_available_tools()

            async for block in provider_client.generate_stream(
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
                match block.type:
                    case StreamBlockType.THINKING:
                        # Show thinking/processing states
                        yield f"_{block.content}_\n"

                    case StreamBlockType.CONTENT:
                        # Regular content
                        full_content.append(block.content)
                        yield block.content

                    case StreamBlockType.TOOL_START:
                        # Tool execution starting
                        tool_msg = f"\n_Using tool: {block.tool_name}_\n"
                        full_content.append(tool_msg)
                        yield tool_msg

                    case StreamBlockType.TOOL_CALL:
                        # Tool being called with args
                        tool_call_msg = f"```json\n{block.tool_args}\n```\n"
                        full_content.append(tool_call_msg)
                        yield tool_call_msg

                    case StreamBlockType.TOOL_RESULT:
                        if isinstance(block.content, list):
                            formatted_content = self._format_tool_content_for_display(block.content)
                            tool_result_msg = f"\n_Tool result from {block.tool_name}:_\n{formatted_content}\n"
                        else:
                            tool_result_msg = f"\n_Tool result from {block.tool_name}:_\n```\n{block.content}\n```\n"

                        full_content.append(tool_result_msg)
                        tool_outputs.append(
                            {"tool_name": block.tool_name, "content": block.content, "call_id": block.tool_call_id}
                        )
                        yield tool_result_msg

                    case StreamBlockType.ERROR:
                        # Handle errors
                        error_msg = f"\n**Error**: {block.error_detail}\n"
                        full_content.append(error_msg)
                        yield error_msg
                        return

                    case StreamBlockType.DONE:
                        # Stream completion
                        await self.finalize_assistant_message(
                            chat_session=chat_session,
                            message_id=message_id,
                            model=model,
                            current_message=current_message,
                            full_content="".join(full_content),
                            params=params,
                            provider_client=provider_client,
                            metadata={"tool_outputs": tool_outputs, **block.metadata},
                        )
                        break

        except Exception as error:
            error_msg = await self.handle_provider_error(error)
            yield f"\n**System Error**: {error_msg}\n"

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
        Generate complete response.
        """
        try:
            # Get conversation history
            history = await self.get_conversation_history(
                chat_session=chat_session,
                current_message_id=user_message.id,
            )
            model_params = self.get_model_params(model=model, request_params=request.model_params)
            system_context = chat_session.system_context

            # Generate response with history
            content, input_tokens, output_tokens = await provider_client.generate(
                prompt=request.prompt,
                model=model.name,
                system_context=system_context,
                messages=history,
                **model_params,
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
                    "max_tokens": request.model_params.max_tokens,
                    "temperature": request.model_params.temperature,
                    "top_p": request.model_params.top_p,
                },
            )

            return CompletionResponse(
                content=content,
                model=model.name,
                provider=provider_client.provider.name,
                usage=ChatUsage(**assistant_message.get_usage()),
            )
        except Exception as error:
            error_message = await self.handle_provider_error(error)
            return CompletionResponse(
                content=error_message,
                model=model.name,
                provider=provider_client.provider.type.value,
                usage=ChatUsage(input_tokens=0, output_tokens=0, input_cost=0, output_cost=0, total_cost=0),
            )
