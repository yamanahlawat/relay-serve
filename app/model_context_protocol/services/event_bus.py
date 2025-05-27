import asyncio
from typing import Any, Callable

from loguru import logger


class MCPEventBus:
    """
    Event bus for MCP components to communicate without direct dependencies.

    This provides a publish-subscribe pattern that decouples components
    and avoids circular dependencies while maintaining clear communication.

    Example usage:
    ```python
    # Publishing an event
    await mcp_event_bus.publish("mcp_server_started", {"server_name": "my-server"})

    # Subscribing to an event
    async def handle_server_shutdown(data: dict[str, Any]) -> None:
        server_name = data.get("server_name")
        print(f"Server {server_name} was shut down")

    await mcp_event_bus.subscribe("mcp_server_shutdown", handle_server_shutdown)
    ```
    """

    def __init__(self) -> None:
        # Event subscribers
        self._subscribers: dict[str, list[Callable]] = {}
        # Lock for thread-safe subscriber management
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for a specific event type.

        Args:
            event_type: Type of event to subscribe to (e.g., "mcp_server_started")
            callback: Async function to call when event is published. Can accept
                      a data dictionary or no arguments.
        Example:
            ```python
            async def on_server_started(data):
                print(f"Server {data['server_name']} started")

            await event_bus.subscribe("mcp_server_started", on_server_started)
            ```
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to event {event_type}")

    async def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Remove a callback for a specific event type.
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove (must be the same function reference)
        Note:
            If the callback doesn't exist, this is a no-op (no error is raised).
        """
        async with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from event {event_type}")

    async def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """
        Publish an event to all subscribers.
        Args:
            event_type: Type of event to publish
            data: Event payload data as a dictionary
        Example:
            ```python
            # Publishing server startup with metadata
            await event_bus.publish("mcp_server_started", {
                "server_name": "my-server",
                "startup_time_ms": 1250
            })
            ```
        Note:
            If a subscriber raises an exception, it's logged but not propagated.
            This ensures one failing handler doesn't prevent others from executing.
        """
        logger.debug(f"Publishing event {event_type} with data: {data}")

        if event_type not in self._subscribers:
            return

        # Create a copy of subscribers to avoid issues if subscribers change during iteration
        subscribers = list(self._subscribers[event_type])

        # Execute all callbacks
        for callback in subscribers:
            try:
                if data:
                    await callback(data)
                else:
                    await callback()
            except Exception as e:
                # Log but don't propagate callback errors
                logger.error(f"Error in event handler for {event_type}: {e}")


# Create a singleton instance
mcp_event_bus = MCPEventBus()
