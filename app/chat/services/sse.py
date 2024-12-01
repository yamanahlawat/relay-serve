from typing import AsyncGenerator
from uuid import UUID

from fastapi import BackgroundTasks
from loguru import logger
from redis.asyncio import Redis

from app.chat.schemas.stream import StreamEvent, StreamResponse
from app.core.config import settings


class SSEConnectionManager:
    """
    SSE Connection Manager using Redis Pub/Sub for distributed message broadcasting.
    """

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @classmethod
    async def create(cls) -> "SSEConnectionManager":
        """
        Factory method to create connection manager with Redis.
        """
        redis = Redis.from_url(
            url=str(settings.REDIS.DSN),
            encoding="utf-8",
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
        )
        return cls(redis=redis)

    async def connect(self, session_id: UUID) -> None:
        """
        Ensure a session key exists in Redis for tracking TTL.
        """
        session_key = f"sse:session:{session_id}"
        await self.redis.setex(session_key, 3600, "active")  # Set a 1-hour TTL

    async def disconnect(self, session_id: UUID) -> None:
        """
        Cleanup session-specific keys when the connection is terminated.
        """
        session_key = f"sse:session:{session_id}"
        await self.redis.delete(session_key)

    async def stream_generator(
        self,
        session_id: UUID,
        generator: AsyncGenerator[str, None],
        background_tasks: BackgroundTasks,
    ) -> AsyncGenerator[str, None]:
        """
        Streams data for the session using Redis Pub/Sub.
        """
        session_key = f"sse:session:{session_id}"
        pubsub_channel = f"sse:stream:{session_id}"

        try:
            logger.info(f"Starting stream for session {session_id}")
            # Ensure the session is active
            await self.connect(session_id=session_id)

            # Publish messages to the channel as they are generated
            async for chunk in generator:
                if not await self.redis.exists(session_key):
                    break
                await self.redis.publish(pubsub_channel, chunk)
                yield chunk

        except Exception as error:
            response = StreamResponse(event=StreamEvent.ERROR, error=str(error))
            yield f"data: {response.model_dump_json()}\n\n"

        finally:
            # Cleanup the session on disconnect
            background_tasks.add_task(self.disconnect, session_id)

    async def cleanup(self) -> None:
        """
        Gracefully shutdown Redis connections.
        """
        await self.redis.close()


# Initialize manager
manager = None


async def get_sse_manager() -> SSEConnectionManager:
    """
    Get or create SSE manager instance.
    """
    global manager
    if manager is None:
        manager = await SSEConnectionManager.create()
    return manager
