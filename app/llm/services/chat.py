import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

import aiofiles
from llm_registry import CapabilityRegistry, ModelCapabilities
from llm_registry.exceptions import ModelNotFoundError
from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent, AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl
from pydantic_ai.exceptions import (
    AgentRunError,
    FallbackExceptionGroup,
    ModelRetry,
    UserError,
)
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
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from sqlalchemy.exc import SQLAlchemyError

from app.attachment.constants import AttachmentType
from app.core.config import settings
from app.core.database.session import AsyncSessionLocal
from app.core.storage.utils import get_attachment_download_url
from app.llm.providers.factory import ProviderFactory
from app.llm.services.stream_block_factory import StreamBlockFactory
from app.llm.services.tool_tracker import ToolCallTracker
from app.mcp_server.lifecycle import mcp_lifecycle_manager
from app.message.constants import MessageRole, MessageStatus
from app.message.model import ChatMessage
from app.message.schema import MessageCreate, MessageRead, MessageUpdate, MessageUsage
from app.message.service import ChatMessageService
from app.model.model import LLMModel
from app.provider.model import LLMProvider

# Constants
ONE_MILLION_TOKENS = 1_000_000


class ChatService:
    """
    Service for handling chat completions with pydantic_ai
    """

    # Attachment type mapping
    _ATTACHMENT_TYPE_MAP = {
        AttachmentType.IMAGE: ImageUrl,
        AttachmentType.VIDEO: VideoUrl,
        AttachmentType.AUDIO: AudioUrl,
        AttachmentType.DOCUMENT: DocumentUrl,
    }

    def __init__(self) -> None:
        """Initialize the chat service with database session."""
        self.model_registry = CapabilityRegistry()

    async def _create_agent(
        self,
        provider: LLMProvider,
        model: LLMModel,
        toolsets: list,
        system_prompt: str | None = None,
    ) -> Agent:
        """Create an agent for the provider with MCP tools."""
        return ProviderFactory.create_agent(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            toolsets=toolsets,
        )

    async def _process_single_attachment(
        self, attachment, is_localhost: bool
    ) -> BinaryContent | ImageUrl | VideoUrl | AudioUrl | DocumentUrl | None:
        """
        Process a single attachment to pydantic_ai compatible format.
        """
        if not attachment.storage_path:
            return None

        # Handle localhost case with binary content
        if is_localhost:
            file_path = Path(attachment.storage_path)
            if file_path.exists():
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read()
                return BinaryContent(data=content, media_type=attachment.mime_type)
            return None

        # Handle remote URLs
        attachment_url = get_attachment_download_url(storage_path=attachment.storage_path)
        attachment_class = self._ATTACHMENT_TYPE_MAP.get(attachment.type)

        if attachment_class:
            # Use storage_path for VIDEO/AUDIO, attachment_url for IMAGE/DOCUMENT
            url = (
                attachment.storage_path
                if attachment.type in (AttachmentType.VIDEO, AttachmentType.AUDIO)
                else attachment_url
            )
            return attachment_class(url=url)
        else:
            logger.warning(f"Unsupported attachment: ID: {attachment.id}: Type: {attachment.type}")
            return None

    async def _convert_attachments_to_pydantic(
        self, message: ChatMessage
    ) -> list[BinaryContent | ImageUrl | VideoUrl | AudioUrl | DocumentUrl] | None:
        """
        Convert message attachments to pydantic_ai compatible formats.
        """
        if not message.direct_attachments:
            return None

        is_localhost = "localhost" in str(settings.BASE_URL)

        attachment_tasks = [
            self._process_single_attachment(attachment, is_localhost) for attachment in message.direct_attachments
        ]

        results = await asyncio.gather(*attachment_tasks, return_exceptions=True)

        data = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error processing attachment: {result}")
                continue
            if result is not None:
                data.append(result)

        return data if data else None

    async def _prepare_message_history(self, session_id: UUID, current_message: ChatMessage) -> list[ModelMessage]:
        """
        Prepare message history with recent messages from the session.
        """
        try:
            async with AsyncSessionLocal() as db:
                message_service = ChatMessageService(db=db)
                recent_messages = await message_service.get_session_context(
                    session_id=session_id,
                    exclude_message_id=current_message.id if current_message else None,
                )

                # Convert database messages to ModelMessage objects
                message_history = []
                for msg in recent_messages or []:
                    content = msg.content or ""
                    if msg.role == MessageRole.USER.value:
                        message_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
                    elif msg.role == MessageRole.ASSISTANT.value:
                        message_history.append(ModelResponse(parts=[TextPart(content=content)]))

                return message_history

        except (SQLAlchemyError, AttributeError, TypeError) as e:
            logger.warning(f"Error retrieving message history: {e}")
            return []

    def _prepare_model_settings(
        self,
        model: LLMModel,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None = None,
        model_capability: ModelCapabilities | None = None,
    ) -> ModelSettings | None:
        """
        Prepare model settings from parameters and defaults.
        """
        settings_dict = {}

        # Set temperature
        if model_capability and not model_capability.api_params.temperature:
            logger.warning(f"Model {model.name} does not support temperature parameter, ignoring temperature setting")
        elif temperature is not None:
            settings_dict["temperature"] = temperature
        elif model.default_temperature is not None:
            settings_dict["temperature"] = model.default_temperature

        # Set max tokens
        if model_capability and not model_capability.api_params.max_tokens:
            logger.warning(f"Model {model.name} does not support max_tokens parameter, ignoring max_tokens setting")
        elif max_tokens is not None:
            settings_dict["max_tokens"] = max_tokens
        elif model.default_max_tokens is not None:
            settings_dict["max_tokens"] = model.default_max_tokens

        # Set top_p
        if model_capability and not model_capability.api_params.top_p:
            logger.warning(f"Model {model.name} does not support top_p parameter, ignoring top_p setting")
        elif top_p is not None:
            settings_dict["top_p"] = top_p
        elif model.default_top_p is not None:
            settings_dict["top_p"] = model.default_top_p

        return ModelSettings(**settings_dict) if settings_dict else None

    async def _update_message_status(
        self, session_id: UUID, message_id: UUID, status: MessageStatus, extra_data: dict | None = None
    ) -> None:
        """
        Update message status in database.
        """
        try:
            async with AsyncSessionLocal() as db:
                message_service = ChatMessageService(db=db)
                update_data = MessageUpdate(status=status)
                if extra_data:
                    update_data.extra_data = extra_data
                await message_service.update_message(
                    session_id=session_id,
                    message_id=message_id,
                    message_in=update_data,
                )
        except Exception:
            # Ignore database errors during status updates
            pass

    async def stream_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        session_id: UUID,
        message_id: UUID,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response for an existing message using pydantic_ai.
        Args:
            provider: LLM provider to use
            model: LLM model to use
            session_id: UUID of the chat session
            message_id: UUID of existing message to complete
            system_prompt: Optional system prompt override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            top_p: Optional top_p override
        Yields:
            JSON-serialized StreamBlock objects containing rich streaming information
        Raises:
            ValueError: If message not found or invalid
            RuntimeError: If database or AI operation fails
        """
        # Initialize tool call tracker and stream block collection
        tool_tracker = ToolCallTracker()
        stream_blocks: list[dict[str, Any]] = []

        def collect_and_yield_block(block) -> str:
            """Helper to collect stream blocks and yield JSON"""
            # Store all blocks except ephemeral UI thinking blocks
            # reasoning blocks ARE stored (they contain actual model reasoning)
            if block.type != "thinking":
                stream_blocks.append(block.model_dump())
            # Yield all blocks for streaming (thinking + reasoning + content)
            return block.model_dump_json()

        try:
            initial_block = StreamBlockFactory.create_thinking_block("Processing your request...")
            yield collect_and_yield_block(initial_block)

            asyncio.create_task(
                self._update_message_status(
                    session_id=session_id, message_id=message_id, status=MessageStatus.PROCESSING
                )
            )

            async with AsyncSessionLocal() as db:
                message_service = ChatMessageService(db=db)
                current_message = await message_service.get_message(session_id=session_id, message_id=message_id)
                if not current_message or not current_message.content:
                    raise ValueError(f"Message {message_id} not found or has no content")

            async def get_model_capability():
                try:
                    return self.model_registry.get_model(model_id=model.name)
                except ModelNotFoundError:
                    logger.warning(f"Model {model.name} not found in registry, using default settings")
                    return None

            toolsets_task = mcp_lifecycle_manager.get_running_servers()
            message_history_task = self._prepare_message_history(session_id=session_id, current_message=current_message)
            attachment_task = self._convert_attachments_to_pydantic(current_message)
            model_capability_task = get_model_capability()

            toolsets, message_history, attachment_messages, model_capability = await asyncio.gather(
                toolsets_task, message_history_task, attachment_task, model_capability_task, return_exceptions=True
            )

            if isinstance(toolsets, Exception):
                logger.warning(f"Error getting MCP servers: {toolsets}")
                toolsets = []
            if isinstance(message_history, Exception):
                logger.warning(f"Error retrieving message history: {message_history}")
                message_history = []
            if isinstance(attachment_messages, Exception):
                logger.warning(f"Error processing attachments: {attachment_messages}")
                attachment_messages = None
            if isinstance(model_capability, Exception):
                logger.warning(f"Error getting model capability: {model_capability}")
                model_capability = None

            logger.debug(f"Retrieved {len(toolsets)} running MCP servers for agent")

            agent = await self._create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                toolsets=toolsets,
            )

            model_settings = self._prepare_model_settings(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                model_capability=model_capability,
            )
            if attachment_messages:
                user_prompt = [current_message.content, *attachment_messages]
            else:
                user_prompt = [current_message.content]

            # Use pydantic_ai's rich streaming with message history and current user prompt
            # No need to start/stop servers - they're managed by the lifecycle manager
            async with agent.iter(
                user_prompt=user_prompt,
                message_history=message_history,
                model_settings=model_settings,
            ) as run:
                async for node in run:
                    if agent.is_user_prompt_node(node):
                        # User prompt node - show processing message
                        thinking_block = StreamBlockFactory.create_thinking_block("Understanding your request...")
                        yield collect_and_yield_block(thinking_block)

                    elif agent.is_model_request_node(node):
                        # Model request node - show response generation
                        thinking_block = StreamBlockFactory.create_thinking_block("Thinking about your request...")
                        yield collect_and_yield_block(thinking_block)

                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if isinstance(event, PartStartEvent):
                                    if isinstance(event.part, ThinkingPart):
                                        # Reasoning model thinking - capture actual model reasoning
                                        reasoning_content = getattr(event.part, "content", "")
                                        if reasoning_content:
                                            reasoning_block = StreamBlockFactory.create_reasoning_block(
                                                content=reasoning_content
                                            )
                                            yield collect_and_yield_block(reasoning_block)

                                    elif isinstance(event.part, ToolCallPart):
                                        # Tool call starting - show thinking and tool info
                                        tool_name = getattr(event.part, "tool_name", "unknown")
                                        tool_call_id = getattr(event.part, "tool_call_id", f"part_{event.index}")

                                        # Start tracking this tool call with part index mapping
                                        tool_tracker.start_tool_call(tool_call_id, tool_name, event.index)

                                        # Show user-friendly thinking message for any MCP tool
                                        thinking_block = StreamBlockFactory.create_thinking_block(
                                            f"Let me use {tool_name} to help with that..."
                                        )
                                        yield collect_and_yield_block(thinking_block)

                                        # Show tool call start
                                        tool_start_block = StreamBlockFactory.create_tool_start_block(
                                            tool_name=tool_name,
                                            tool_call_id=tool_call_id,
                                        )
                                        yield collect_and_yield_block(tool_start_block)

                                    elif isinstance(event.part, TextPart):
                                        # Text response starting - yield the initial content
                                        text_content = getattr(event.part, "content", "")
                                        if text_content:
                                            yield StreamBlockFactory.create_text_delta_block(
                                                text_content
                                            ).model_dump_json()

                                elif isinstance(event, PartDeltaEvent):
                                    if isinstance(event.delta, ThinkingPartDelta):
                                        # Streaming reasoning content as it's generated
                                        content_delta = getattr(event.delta, "content_delta", "")
                                        if content_delta:
                                            reasoning_delta = StreamBlockFactory.create_reasoning_block(
                                                content=content_delta
                                            )
                                            yield collect_and_yield_block(reasoning_delta)

                                    elif isinstance(event.delta, TextPartDelta):
                                        # Text content delta
                                        content = event.delta.content_delta
                                        if content:
                                            yield StreamBlockFactory.create_text_delta_block(content).model_dump_json()

                                    elif isinstance(event.delta, ToolCallPartDelta):
                                        # Tool call arguments being built - stream raw delta chunks
                                        args_delta = event.delta.args_delta
                                        if args_delta:
                                            # Get the tool call ID using part index mapping
                                            tool_call_id = tool_tracker.get_tool_call_id_by_part_index(event.index)
                                            if tool_call_id:
                                                # Get tool info for the args delta block
                                                tool_info = tool_tracker.get_tool_info(tool_call_id)
                                                tool_name = (
                                                    tool_info.get("tool_name", "unknown") if tool_info else "unknown"
                                                )

                                                # Create and stream the args delta block with raw delta
                                                args_delta_block = StreamBlockFactory.create_tool_args_delta_block(
                                                    tool_name=tool_name,
                                                    tool_call_id=tool_call_id,
                                                    args_delta=str(args_delta),
                                                )
                                                yield collect_and_yield_block(args_delta_block)

                                elif isinstance(event, FinalResultEvent):
                                    # Final result from model - show completion
                                    thinking_block = StreamBlockFactory.create_final_result_event_block(
                                        tool_name=event.tool_name
                                    )
                                    yield collect_and_yield_block(thinking_block)

                    elif agent.is_call_tools_node(node):
                        # Tool execution node - show tool calls and results
                        thinking_block = StreamBlockFactory.create_call_tools_node_start_block()
                        yield collect_and_yield_block(thinking_block)

                        async with node.stream(ctx=run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    # Tool is being called - show complete call info
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

                                    # Show the tool call
                                    tool_call_block = StreamBlockFactory.create_function_tool_call_event_block(
                                        tool_name=event.part.tool_name,
                                        tool_call_id=event.part.tool_call_id,
                                        tool_args=tool_args,
                                    )
                                    yield collect_and_yield_block(tool_call_block)

                                elif isinstance(event, FunctionToolResultEvent):
                                    # Tool result received - show result and interpretation
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

                                    # Show tool result
                                    tool_result_block = StreamBlockFactory.create_function_tool_result_event_block(
                                        tool_call_id=event.tool_call_id,
                                        tool_name=tool_name,
                                        result_content=result_content,
                                    )
                                    yield collect_and_yield_block(tool_result_block)

                                    # Show user-friendly interpretation
                                    interpretation = f"Got some helpful information from {tool_name}"
                                    interpretation_block = StreamBlockFactory.create_thinking_block(interpretation)
                                    yield collect_and_yield_block(interpretation_block)

                                    # Clean up tool tracking for completed call
                                    tool_tracker.cleanup_tool_call(event.tool_call_id)

                    elif agent.is_end_node(node):
                        # Agent run complete - will send final message block after streaming
                        if run.result and run.result.output:
                            assert run.result.output == node.data.output

            # Save AI response to database after streaming completes
            final_output = run.result.output if run.result else None
            if final_output and str(final_output).strip():
                usage_data = run.result.usage() if run.result else None

                # Create message with complete content, usage information, and stream blocks
                assistant_message = MessageCreate(
                    content=str(final_output).strip(),
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    parent_id=message_id,
                    extra_data={
                        "stream_blocks": stream_blocks,
                    },
                )

                # Add usage data if available
                if usage_data:
                    costs = self._calculate_cost(
                        model=model,
                        input_tokens=getattr(usage_data, "request_tokens", 0),
                        output_tokens=getattr(usage_data, "response_tokens", 0),
                        model_capability=model_capability,
                    )
                    assistant_message.usage = MessageUsage(
                        input_tokens=getattr(usage_data, "request_tokens", 0),
                        output_tokens=getattr(usage_data, "response_tokens", 0),
                        input_cost=costs["input_cost"],
                        output_cost=costs["output_cost"],
                    )

                # Save the complete message to database for persistence
                async with AsyncSessionLocal() as db:
                    message_service = ChatMessageService(db=db)
                    created_message = await message_service.create_message(
                        message_in=assistant_message,
                        session_id=session_id,
                    )

                # Send final message block with the persisted message data and usage
                final_block = StreamBlockFactory.create_done_block(content=final_output)
                final_block.message = MessageRead.model_validate(created_message)
                final_block.usage = assistant_message.usage.model_dump() if assistant_message.usage else None
                yield collect_and_yield_block(final_block)

            # Update original message status to completed
            await self._update_message_status(session_id, message_id, MessageStatus.COMPLETED)
        except ValidationError as error:
            logger.error(f"Validation error in stream_response: {error}")
            raise ValueError(f"Invalid input data: {error}") from error
        except SQLAlchemyError as error:
            logger.error(f"Database error in stream_response: {error}")
            raise RuntimeError(f"Database operation failed: {error}") from error
        except (AgentRunError, UserError, ModelRetry, FallbackExceptionGroup) as error:
            logger.error(f"AI error in stream_response: {error}")
            raise
        finally:
            # Clean up tool tracker state after streaming completes
            if "tool_tracker" in locals():
                tool_tracker.reset()

            # Update message status to failed if we're in an exception context
            if sys.exc_info()[0] is not None:
                await self._update_message_status(
                    session_id=session_id, message_id=message_id, status=MessageStatus.FAILED
                )

    def _calculate_cost(
        self, model: LLMModel, input_tokens: int, output_tokens: int, model_capability: ModelCapabilities | None = None
    ) -> dict[str, float]:
        """
        Calculate input, output, and total costs based on model pricing.
        """
        if not model_capability or not model_capability.token_costs:
            logger.warning(f"Model {model.name} does not have token costs defined")
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        costs = model_capability.token_costs
        input_cost = (input_tokens / ONE_MILLION_TOKENS) * costs.input_cost
        output_cost = (output_tokens / ONE_MILLION_TOKENS) * costs.output_cost
        total_cost = input_cost + output_cost
        return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": total_cost}
