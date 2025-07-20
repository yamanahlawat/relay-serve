"""
MCP Server Lifecycle Management Service.

This service manages the lifecycle of MCP servers at the application level,
ensuring servers are started once during app startup and reused across
all chat sessions and agent interactions.

Uses pydantic-ai's native lifecycle management patterns with AsyncExitStack.
"""

import asyncio
from contextlib import AsyncExitStack

from loguru import logger
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.services.domain import MCPServerDomainService


class MCPServerLifecycleManager:
    """
    Manages the lifecycle of MCP servers using pydantic-ai's native patterns.

    This singleton service:
    - Uses AsyncExitStack for proper resource management
    - Leverages pydantic-ai's built-in reference counting
    - Handles graceful startup and shutdown
    """

    def __init__(self) -> None:
        """Initialize the lifecycle manager."""
        self._servers: list[MCPServerStdio | MCPServerStreamableHTTP] = []
        self._exit_stack: AsyncExitStack | None = None
        self._started = False
        self._lock = asyncio.Lock()

    async def start_enabled_servers(self) -> None:
        """
        Start all enabled MCP servers using pydantic-ai's lifecycle patterns.

        Uses AsyncExitStack for proper resource management.
        """
        async with self._lock:
            if self._started:
                logger.warning("MCP servers are already started")
                return

            self._exit_stack = AsyncExitStack()

            try:
                # Get servers from database
                async with AsyncSessionLocal() as db:
                    mcp_service = MCPServerDomainService(db=db)
                    servers = await mcp_service.get_mcp_servers_for_agent()

                if not servers:
                    logger.info("No MCP servers configured")
                    self._started = True
                    return

                logger.info(f"Starting {len(servers)} MCP servers...")

                # Start servers using pydantic-ai's context manager pattern
                started_servers = []
                for i, server in enumerate(servers):
                    try:
                        # Let pydantic-ai handle the lifecycle through AsyncExitStack
                        await self._exit_stack.enter_async_context(server)
                        started_servers.append(server)

                        # Get server identifier for logging
                        server_name = getattr(server, "command", getattr(server, "url", f"server_{i}"))
                        logger.debug(f"Started MCP server: {server_name}")

                    except Exception as e:
                        logger.error(f"Failed to start MCP server {i}: {e}")
                        # Continue with other servers rather than failing completely
                        continue

                self._servers = started_servers
                self._started = True

                logger.info(f"Successfully started {len(self._servers)}/{len(servers)} MCP servers")

            except Exception as e:
                logger.error(f"Failed to start MCP servers: {e}")
                await self._cleanup()
                raise

    async def get_running_servers(self) -> list[MCPServerStdio | MCPServerStreamableHTTP]:
        """
        Get all currently running MCP servers.

        Returns:
            List of running MCP server instances ready for agent use
        """
        async with self._lock:
            if not self._started:
                logger.warning("MCP servers are not started yet. Call start_enabled_servers() first.")
                return []

            # Return copy of servers - pydantic-ai manages server state internally
            return self._servers.copy()

    async def shutdown(self) -> None:
        """
        Shutdown all running MCP servers using pydantic-ai's lifecycle management.
        """
        async with self._lock:
            if not self._started:
                logger.debug("MCP servers are not started, nothing to shutdown")
                return

            logger.info(f"Shutting down {len(self._servers)} MCP servers...")

            await self._cleanup()
            logger.info("All MCP servers have been shut down")

    async def _cleanup(self) -> None:
        """
        Internal cleanup using AsyncExitStack for proper resource management.
        """
        try:
            if self._exit_stack:
                # Let AsyncExitStack handle proper shutdown of all servers
                await self._exit_stack.aclose()
                self._exit_stack = None
        except Exception as e:
            logger.error(f"Error during server cleanup: {e}")
        finally:
            # Reset state
            self._servers.clear()
            self._started = False


# Global singleton instance
mcp_lifecycle_manager = MCPServerLifecycleManager()
