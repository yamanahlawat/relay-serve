"""Memory management service using mem0."""

from typing import Any

from mem0 import AsyncMemory

from app.core.config import settings


class MemoryService:
    """Service for managing conversation memory using mem0."""

    def __init__(self) -> None:
        """Initialize the memory service."""
        self._memory: AsyncMemory | None = None

    @property
    async def memory(self) -> AsyncMemory:
        """Get or create the memory instance."""
        if self._memory is None:
            self._memory = await self._create_memory()
        return self._memory

    async def _create_memory(self) -> AsyncMemory:
        """Create and configure the mem0 memory instance."""
        config = {
            "vector_store": {
                "provider": settings.MEM0.VECTOR_STORE_PROVIDER,
                "config": {
                    "host": settings.MEM0.VECTOR_STORE_HOST,
                    "port": settings.MEM0.VECTOR_STORE_PORT,
                    "collection_name": settings.MEM0.VECTOR_STORE_COLLECTION_NAME,
                },
            },
            "llm": {
                "provider": settings.MEM0.MEMORY_LLM_PROVIDER,
                "config": {
                    "model": settings.MEM0.MEMORY_LLM_MODEL,
                    "temperature": settings.MEM0.MEMORY_LLM_TEMPERATURE,
                },
            },
            "embedder": {
                "provider": settings.MEM0.EMBEDDER_PROVIDER,
                "config": {
                    "model": settings.MEM0.EMBEDDER_MODEL,
                    "embedding_dims": settings.MEM0.EMBEDDER_DIMENSIONS,
                },
            },
        }

        # Add graph store if enabled
        if settings.MEM0.ENABLE_GRAPH_MEMORY:
            config["graph_store"] = {
                "provider": settings.MEM0.GRAPH_STORE_PROVIDER,
                "config": {
                    "url": settings.MEM0.GRAPH_STORE_URL,
                    "username": settings.MEM0.GRAPH_STORE_USERNAME,
                    "password": (
                        settings.MEM0.GRAPH_STORE_PASSWORD.get_secret_value()
                        if settings.MEM0.GRAPH_STORE_PASSWORD
                        else None
                    ),
                },
            }

        # Initialize AsyncMemory with the same config structure as Memory
        return await AsyncMemory.from_config(config_dict=config)

    async def add_memory(
        self,
        message: str,
        user_id: str,
        session_id: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a message to memory and return the memory ID."""
        memory_instance = await self.memory
        result = await memory_instance.add(
            messages=[{"role": role, "content": message}],
            user_id=user_id,
            agent_id=session_id,  # Use agent_id for session grouping
            metadata=metadata,
        )
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("id", "")
        return ""

    async def get_memories(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get memories for a user and optionally a specific session."""
        try:
            memory_instance = await self.memory
            memories = await memory_instance.get_all(user_id=user_id, agent_id=session_id)
            if isinstance(memories, dict) and "results" in memories:
                return memories["results"][:limit]
            # Handle case where memories is returned as a list directly
            return []
        except Exception:
            return []

    async def search_memories(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memories for a user and optionally a specific session."""
        try:
            memory_instance = await self.memory
            results = await memory_instance.search(query=query, user_id=user_id, agent_id=session_id)
            if isinstance(results, dict) and "results" in results:
                return results["results"][:limit]
            # Handle case where results is returned as a list directly
            return []
        except Exception:
            return []

    async def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get conversation history for a specific session."""
        try:
            memory_instance = await self.memory
            # Use get_all with agent_id filter since history() only takes memory_id
            memories = await memory_instance.get_all(user_id=user_id, agent_id=session_id)
            if isinstance(memories, dict) and "results" in memories:
                return memories["results"][:limit]
            # Handle case where memories is returned as a list directly
            return []
        except Exception:
            return []

    async def update_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a specific memory."""
        memory_instance = await self.memory
        # Use the data parameter as required by mem0
        update_data: dict[str, Any] = {"text": text}
        if metadata:
            update_data.update(metadata)
        await memory_instance.update(memory_id=memory_id, data=update_data)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a specific memory."""
        memory_instance = await self.memory
        await memory_instance.delete(memory_id=memory_id)

    async def delete_session_memories(self, session_id: str, user_id: str) -> None:
        """Delete all memories for a specific session."""
        memory_instance = await self.memory
        await memory_instance.delete_all(user_id=user_id, agent_id=session_id)

    async def delete_user_memories(self, user_id: str) -> None:
        """Delete all memories for a specific user."""
        memory_instance = await self.memory
        await memory_instance.delete_all(user_id=user_id)


memory_service = MemoryService()
