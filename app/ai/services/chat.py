import json
from typing import Any, AsyncIterator
from uuid import UUID

from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers.factory import ProviderFactory
from app.ai.schemas.chat import TransparencySettings
from app.ai.services.stream_block_factory import StreamBlockFactory
from app.ai.services.tool_tracker import ToolCallTracker
from app.chat.constants import MessageRole, MessageStatus
from app.chat.schemas.message import MessageCreate, MessageRead, MessageUpdate, MessageUsage
from app.chat.services.message import ChatMessageService
from app.chat.services.session import ChatSessionService
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider
from app.model_context_protocol.services.domain import MCPServerDomainService


class ChatService:
    """Service for handling chat completions with pydantic_ai"""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the chat service with database session."""
        self.db = db
        self._agents: dict[str, Agent] = {}
        self.message_service = ChatMessageService(db=db)
        self.session_service = ChatSessionService(db=db)
        # Initialize MCP domain service for server management
        self.mcp_service = MCPServerDomainService(db=db)

    def _get_agent_key(self, provider: LLMProvider, model: LLMModel) -> str:
        """Generate a unique key for caching agents."""
        return f"{provider.type.value}:{model.name}"

    async def _get_mcp_servers_for_agent(self) -> list:
        """Get MCP servers for pydantic-ai agent."""
        try:
            mcp_servers = await self.mcp_service.get_mcp_servers_for_agent()
            logger.debug(f"Retrieved {len(mcp_servers)} MCP servers for agent")
            return mcp_servers
        except Exception as e:
            logger.warning(f"Failed to get MCP servers: {e}")
            return []

    async def _get_or_create_agent(
        self,
        provider: LLMProvider,
        model: LLMModel,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Get or create a cached agent for the provider with MCP tools."""
        cache_key = self._get_agent_key(provider, model)

        if cache_key not in self._agents:
            # Get MCP servers for the agent
            mcp_servers = await self._get_mcp_servers_for_agent()

            logger.debug(f"Creating agent with {len(tools or [])} manual tools and {len(mcp_servers)} MCP servers")

            self._agents[cache_key] = ProviderFactory.create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                mcp_servers=mcp_servers,
            )

        return self._agents[cache_key]

    async def _prepare_message_history(
        self,
        session_id: UUID,
        current_message_id: UUID | None = None,
    ) -> list[ModelMessage]:
        """
        Prepare message history with recent messages from the session.
        Returns a list of ModelMessage objects for pydantic_ai.
        """
        message_history: list[ModelMessage] = []

        # Get recent conversation messages for context
        try:
            recent_messages = await self.message_service.get_session_context(
                session_id=session_id,
                exclude_message_id=current_message_id,
            )

            # Convert database messages to ModelMessage objects
            if recent_messages:
                for msg in reversed(recent_messages):  # Reverse to get chronological order
                    content = msg.content or ""
                    if msg.role == MessageRole.USER.value:
                        message_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
                    elif msg.role == MessageRole.ASSISTANT.value:
                        message_history.append(ModelResponse(parts=[TextPart(content=content)]))

        except SQLAlchemyError as e:
            logger.warning(f"Database error retrieving message history: {e}")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Data formatting error in message history: {e}")

        return message_history

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
        transparency: TransparencySettings | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response for an existing message using pydantic_ai with rich streaming.

        Args:
            provider: LLM provider to use
            model: LLM model to use
            session_id: UUID of the chat session
            message_id: UUID of existing message to complete
            system_prompt: Optional system prompt override
            tools: Optional tools for the agent
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            transparency: Optional transparency settings for controlling stream visibility

        Yields:
            JSON-serialized StreamBlock objects containing rich streaming information

        Raises:
            ValueError: If message not found or invalid
            RuntimeError: If database or AI operation fails
        """
        # Initialize transparency settings with defaults
        transparency = transparency or TransparencySettings()

        # Initialize tool call tracker for managing tool state
        tool_tracker = ToolCallTracker()
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
            agent = await self._get_or_create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Prepare message history
            message_history = await self._prepare_message_history(session_id=session_id, current_message_id=message_id)

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

            model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else None

            # Use pydantic_ai's rich streaming with message history and current user prompt
            async with agent.run_mcp_servers():
                async with agent.iter(
                    user_prompt=message_content,
                    message_history=message_history,
                    model_settings=model_settings,
                ) as run:
                    async for node in run:
                        if agent.is_user_prompt_node(node):
                            # User prompt node - show what we're processing
                            if transparency.show_thinking:
                                yield StreamBlockFactory.create_thinking_block(
                                    "Processing your request..."
                                ).model_dump_json()

                        elif agent.is_model_request_node(node):
                            # Model request node - stream partial request tokens
                            if transparency.show_thinking:
                                yield StreamBlockFactory.create_thinking_block(
                                    "Generating response..."
                                ).model_dump_json()

                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    if isinstance(event, PartStartEvent):
                                        part_type = type(event.part).__name__

                                        if part_type == "ToolCallPart":
                                            # Tool call starting - always show basic tool info
                                            tool_name = getattr(event.part, "tool_name", "unknown")
                                            tool_call_id = getattr(event.part, "tool_call_id", f"part_{event.index}")

                                            # Start tracking this tool call with part index mapping
                                            tool_tracker.start_tool_call(tool_call_id, tool_name, event.index)

                                            if transparency.show_tool_calls:
                                                yield StreamBlockFactory.create_tool_start_block(
                                                    tool_name=tool_name,
                                                    tool_call_id=tool_call_id,
                                                ).model_dump_json()

                                            # Always show part start for tool calls (this is what shows the tool name)
                                            part_info = f"tool_name={tool_name}, tool_call_id={tool_call_id}"
                                            yield StreamBlockFactory.create_part_start_block(
                                                part_index=event.index,
                                                part_type=part_type,
                                                part_info=part_info,
                                            ).model_dump_json()

                                        elif part_type == "TextPart":
                                            if transparency.show_typing_indicators:
                                                # Text response starting - optional typing indicator
                                                yield StreamBlockFactory.create_thinking_block(
                                                    "Typing..."
                                                ).model_dump_json()

                                            # Show detailed part events only if debugging is enabled
                                            if transparency.show_part_events:
                                                part_info = f"content={repr(getattr(event.part, 'content', ''))}"
                                                yield StreamBlockFactory.create_part_start_block(
                                                    part_index=event.index,
                                                    part_type=part_type,
                                                    part_info=part_info,
                                                ).model_dump_json()

                                        else:
                                            # Other part types - only show if debugging is enabled
                                            if transparency.show_part_events:
                                                part_info = repr(event.part)
                                                yield StreamBlockFactory.create_part_start_block(
                                                    part_index=event.index,
                                                    part_type=part_type,
                                                    part_info=part_info,
                                                ).model_dump_json()

                                    elif isinstance(event, PartDeltaEvent):
                                        if isinstance(event.delta, TextPartDelta):
                                            # Text content delta - always stream for real-time display
                                            content = event.delta.content_delta
                                            if content:
                                                yield StreamBlockFactory.create_text_delta_block(
                                                    content
                                                ).model_dump_json()

                                        elif (
                                            isinstance(event.delta, ToolCallPartDelta) and transparency.show_tool_calls
                                        ):
                                            # Tool call arguments being built - show if tool calls enabled
                                            args_delta = event.delta.args_delta
                                            if args_delta:
                                                args_delta_str = (
                                                    str(args_delta) if not isinstance(args_delta, str) else args_delta
                                                )

                                                # Get the actual tool call ID using part index mapping
                                                tool_call_id = tool_tracker.get_tool_call_id_by_part_index(event.index)
                                                if not tool_call_id:
                                                    # Fallback if mapping fails
                                                    tool_call_id = f"part_{event.index}"

                                                # Use tool tracker to accumulate and parse args
                                                current_args = tool_tracker.accumulate_args(
                                                    tool_call_id, args_delta_str
                                                )
                                                tool_info = tool_tracker.get_tool_info(tool_call_id)
                                                tool_name = (
                                                    tool_info.get("tool_name", "unknown") if tool_info else "unknown"
                                                )

                                                yield StreamBlockFactory.create_tool_args_delta_block(
                                                    tool_name=tool_name,
                                                    tool_call_id=tool_call_id,
                                                    args_delta=args_delta_str,
                                                    current_args=current_args,
                                                ).model_dump_json()

                                    elif isinstance(event, FinalResultEvent):
                                        # Final result from model - indicates completion
                                        if transparency.show_thinking:
                                            yield StreamBlockFactory.create_final_result_event_block(
                                                tool_name=event.tool_name
                                            ).model_dump_json()

                        elif agent.is_call_tools_node(node):
                            # Tool execution node - stream tool calls and results with transparency control
                            if transparency.show_thinking:
                                yield StreamBlockFactory.create_call_tools_node_start_block().model_dump_json()

                            async with node.stream(ctx=run.ctx) as handle_stream:
                                async for event in handle_stream:
                                    if isinstance(event, FunctionToolCallEvent) and transparency.show_tool_calls:
                                        # Tool is being called - show complete call info if enabled
                                        tool_args = event.part.args
                                        if isinstance(tool_args, str):
                                            # Try to parse JSON string
                                            try:
                                                tool_args = json.loads(tool_args)
                                            except (json.JSONDecodeError, TypeError):
                                                tool_args = {"raw_args": tool_args}
                                        elif not isinstance(tool_args, dict):
                                            tool_args = {"args": tool_args}

                                        # Mark tool call as completed in tracker
                                        tool_tracker.complete_tool_call(event.part.tool_call_id)

                                        yield StreamBlockFactory.create_function_tool_call_event_block(
                                            tool_name=event.part.tool_name,
                                            tool_call_id=event.part.tool_call_id,
                                            tool_args=tool_args,
                                        ).model_dump_json()

                                    elif isinstance(event, FunctionToolResultEvent) and transparency.show_tool_results:
                                        # Tool result received - show complete result if enabled
                                        result_content = ""
                                        if hasattr(event.result, "content"):
                                            if isinstance(event.result.content, str):
                                                result_content = event.result.content
                                            elif isinstance(event.result.content, list):
                                                result_content = ", ".join(str(item) for item in event.result.content)
                                            else:
                                                result_content = str(event.result.content)
                                        else:
                                            result_content = str(event.result)

                                        # Get tool name from tracker before cleaning up
                                        tool_info = tool_tracker.get_tool_info(event.tool_call_id)
                                        tool_name = tool_info.get("tool_name", "unknown") if tool_info else "unknown"

                                        # Clean up tool tracking for completed call
                                        tool_tracker.cleanup_tool_call(event.tool_call_id)

                                        yield StreamBlockFactory.create_function_tool_result_event_block(
                                            tool_call_id=event.tool_call_id,
                                            tool_name=tool_name,
                                            result_content=result_content,
                                        ).model_dump_json()

                        elif agent.is_end_node(node):
                            # Agent run complete - will send final message block after streaming
                            if run.result and run.result.output:
                                assert run.result.output == node.data.output

            # Clean up tool tracker state after streaming completes
            tool_tracker.reset()

            # Save AI response to database after streaming completes using pydantic_ai's final output
            final_output = run.result.output if run.result else None
            if final_output and str(final_output).strip():
                usage_data = run.result.usage() if run.result else None

                # Create message with complete content and usage information
                assistant_message = MessageCreate(
                    content=str(final_output).strip(),
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    parent_id=message_id,
                )

                # Add usage data if available
                if usage_data:
                    assistant_message.usage = MessageUsage(
                        input_tokens=getattr(usage_data, "request_tokens", 0),
                        output_tokens=getattr(usage_data, "response_tokens", 0),
                        input_cost=self._calculate_input_cost(getattr(usage_data, "request_tokens", 0), model),
                        output_cost=self._calculate_output_cost(getattr(usage_data, "response_tokens", 0), model),
                    )

                # Save the complete message to database for persistence
                created_message = await self.message_service.create_message(
                    message_in=assistant_message,
                    session_id=session_id,
                )

                # Send final message block with the persisted message data and usage
                final_block = StreamBlockFactory.create_content_block(content=final_output)
                final_block.message = MessageRead.model_validate(created_message)
                final_block.usage = assistant_message.usage.model_dump() if assistant_message.usage else None
                yield final_block.model_dump_json()

            # Update original message status to completed
            await self.message_service.update_message(
                session_id=session_id,
                message_id=message_id,
                message_in=MessageUpdate(status=MessageStatus.COMPLETED),
            )

        except ValidationError as e:
            logger.error(f"Validation error in stream_response: {e}", exc_info=True)
            # Clean up tool tracker on error
            tool_tracker.reset()
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            yield StreamBlockFactory.create_error_block(
                error_type="ValidationError",
                error_detail=str(e),
            ).model_dump_json()
            raise ValueError(f"Invalid input data: {e}") from e
        except SQLAlchemyError as e:
            logger.error(f"Database error in stream_response: {e}", exc_info=True)
            # Clean up tool tracker on error
            tool_tracker.reset()
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            yield StreamBlockFactory.create_error_block(
                error_type="DatabaseError",
                error_detail=str(e),
            ).model_dump_json()
            raise RuntimeError(f"Database operation failed: {e}") from e
        except ValueError as e:
            logger.error(f"Value error in stream_response: {e}", exc_info=True)
            # Clean up tool tracker on error
            tool_tracker.reset()
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            yield StreamBlockFactory.create_error_block(
                error_type="ValueError",
                error_detail=str(e),
            ).model_dump_json()
            raise
        except Exception as e:
            logger.error(f"Unexpected error streaming response: {e}", exc_info=True)
            # Clean up tool tracker on error
            tool_tracker.reset()
            # Update message status to failed
            try:
                await self.message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=MessageUpdate(status=MessageStatus.FAILED),
                )
            except Exception:
                pass
            yield StreamBlockFactory.create_error_block(
                error_type="UnexpectedError",
                error_detail=str(e),
            ).model_dump_json()
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def _calculate_input_cost(self, input_tokens: int, model: LLMModel) -> float:
        """Calculate input cost based on model pricing."""
        # For now, return 0.0 since the model doesn't have cost fields
        # In the future, add cost fields to LLMModel or use a pricing service
        # TODO: Implement proper cost calculation based on model pricing
        return 0.0

    def _calculate_output_cost(self, output_tokens: int, model: LLMModel) -> float:
        """Calculate output cost based on model pricing."""
        # For now, return 0.0 since the model doesn't have cost fields
        # In the future, add cost fields to LLMModel or use a pricing service
        # TODO: Implement proper cost calculation based on model pricing
        return 0.0

    def _calculate_total_cost(self, usage_data: Any, model: LLMModel) -> float:
        """Calculate total cost from usage data."""
        # For now, return 0.0 since the model doesn't have cost fields
        # In the future, implement proper cost calculation
        # TODO: Implement proper cost calculation based on usage data
        return 0.0


def create_chat_service(db: AsyncSession) -> ChatService:
    """
    Factory function to create a ChatService instance with a database session.
    Args:
        db: Database session
    Returns:
        ChatService instance
    """
    return ChatService(db=db)
