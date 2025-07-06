"""Chat service using pydantic_ai and mem0."""

from typing import Any, AsyncIterator

from pydantic_ai import Agent

from app.ai.memory import memory_service
from app.ai.providers import PydanticAIProviderFactory
from app.llms.models.model import LLMModel
from app.llms.models.provider import LLMProvider


class ChatService:
    """Service for handling chat completions with pydantic_ai and mem0."""

    def __init__(self) -> None:
        """Initialize the chat service."""
        self._agents: dict[str, Agent] = {}

    def _get_agent_key(self, provider: LLMProvider, model: LLMModel) -> str:
        """Generate a unique key for caching agents."""
        return f"{provider.name}:{model.name}"

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
            self._agents[cache_key] = PydanticAIProviderFactory.create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

        return self._agents[cache_key]

    async def _prepare_conversation_context(
        self,
        user_id: str,
        session_id: str,
        message: str,
        include_memory: bool = True,
    ) -> str:
        """Prepare conversation context with memory."""
        context_parts = []

        if include_memory:
            # Search for relevant memories
            relevant_memories = await memory_service.search_memories(
                query=message,
                user_id=user_id,
                session_id=session_id,
                limit=3,
            )

            # Add relevant memories to context
            if relevant_memories:
                memory_context = "\n".join([f"Memory: {memory.get('memory', '')}" for memory in relevant_memories])
                context_parts.append(f"Relevant memories:\n{memory_context}")

        # Add the current user message
        context_parts.append(f"User: {message}")

        return "\n\n".join(context_parts)

    async def generate_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        user_id: str,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        include_memory: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response using pydantic_ai."""
        try:
            # Get or create agent
            agent = self._get_or_create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Prepare conversation context
            conversation_context = await self._prepare_conversation_context(
                user_id=user_id,
                session_id=session_id,
                message=message,
                include_memory=include_memory,
            )

            # Add user message to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=message,
                role="user",
            )

            # Generate response using pydantic_ai
            response = await agent.run(conversation_context)

            # Add AI response to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=str(response.output),
                role="assistant",
            )

            return str(response.output)

        except Exception as e:
            # Log the exception for debugging
            print(f"Error generating response: {e}")
            # Re-raise or handle as needed
            raise

    async def stream_response(
        self,
        provider: LLMProvider,
        model: LLMModel,
        user_id: str,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        include_memory: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response using pydantic_ai, yielding markdown formatted chunks."""
        try:
            # Get or create agent
            agent = self._get_or_create_agent(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Prepare conversation context
            conversation_context = await self._prepare_conversation_context(
                user_id=user_id,
                session_id=session_id,
                message=message,
                include_memory=include_memory,
            )

            # Add user message to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=message,
                role="user",
            )

            # For now, use the regular run method and simulate streaming
            # TODO: Implement proper streaming when pydantic_ai streaming is better understood
            response = await agent.run(conversation_context)
            response_text = str(response.output)

            # Simulate streaming by yielding the response
            yield response_text

            # Add full AI response to memory
            await memory_service.add_memory(
                user_id=user_id,
                session_id=session_id,
                message=response_text,
                role="assistant",
            )

        except Exception as e:
            # Log the exception for debugging
            print(f"Error streaming response: {e}")
            # Re-raise or handle as needed
            raise


chat_service = ChatService()
