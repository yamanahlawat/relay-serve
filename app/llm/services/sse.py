import json
from typing import AsyncGenerator
from uuid import UUID

from fastapi import BackgroundTasks
from loguru import logger
from redis.asyncio import Redis, ConnectionPool

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
        Factory method to create connection manager with Redis connection pool.
        """
        pool = ConnectionPool.from_url(
            url=str(settings.REDIS.DSN),
            encoding="utf-8",
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
            max_connections=20,
            retry_on_timeout=True,
        )
        redis = Redis(connection_pool=pool)
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
        cancel_key = f"sse:cancel:{session_id}"
        await self.redis.delete(session_key)
        await self.redis.delete(cancel_key)  # Remove cancel flag

    async def stop_stream(self, session_id: UUID) -> None:
        """
        Stop an ongoing streaming session by setting a cancellation flag.
        """
        cancel_key = f"sse:cancel:{session_id}"
        await self.redis.set(cancel_key, "1", ex=10)  # Set cancel flag for 10 sec
        logger.info(f"Stop signal sent for session {session_id}")

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
        cancel_key = f"sse:cancel:{session_id}"

        try:
            logger.info(f"Starting stream for session {session_id}")
            # Ensure the session is active
            await self.connect(session_id=session_id)

            # Publish messages to the channel as they are generated
            async for chunk in generator:
                # Use Redis pipeline to batch operations
                pipe = self.redis.pipeline()
                pipe.exists(cancel_key)  # Check for stop signal
                pipe.exists(session_key)  # Check session exists
                pipe.publish(pubsub_channel, chunk)  # Publish chunk

                results = await pipe.execute()
                cancel_exists, session_exists, _ = results

                if cancel_exists:
                    logger.warning(f"Stream cancelled for session {session_id}")
                    break
                if not session_exists:
                    break

                # Format as proper SSE data
                yield f"data: {chunk}\n\n"

        except Exception as error:
            error_message = str(error)
            logger.error(f"Unexpected stream error for session {session_id}: {error_message}")

            response = {"type": "error", "message": error_message}
            yield f"data: {json.dumps(response)}\n\n"

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
