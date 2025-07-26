"""
MCP Server Lifecycle Management Service.

This service manages the lifecycle of MCP servers at the application level,
ensuring servers are started once during app startup and reused across
all chat sessions and agent interactions.

Uses pydantic-ai's native lifecycle management patterns with AsyncExitStack.
"""

import asyncio

from loguru import logger
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.models.server import MCPServer
from app.model_context_protocol.utils import create_server_instance_from_db


class MCPServerLifecycleManager:
    """
    Manages the lifecycle of individual MCP servers using pydantic-ai's native patterns.

    This singleton service:
    - Manages servers individually for granular control
    - Uses AsyncExitStack per server for proper resource management
    - Leverages pydantic-ai's built-in reference counting
    - Handles graceful startup and shutdown
    """

    def __init__(self) -> None:
        """Initialize the lifecycle manager."""
        self._servers: dict[str, MCPServerStdio | MCPServerStreamableHTTP] = {}
        self._server_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start_server(self, server_name: str, server_instance: MCPServerStdio | MCPServerStreamableHTTP) -> bool:
        """
        Start an individual MCP server.

        Args:
            server_name: Name identifier for the server
            server_instance: The MCP server instance to start

        Returns:
            True if started successfully, False otherwise
        """
        async with self._lock:
            if server_name in self._servers:
                logger.warning(f"Server '{server_name}' is already running")
                return True

            try:
                # Create a task to manage the server lifecycle
                async def server_task():
                    async with server_instance:
                        # Keep the server running until task is cancelled
                        try:
                            while True:
                                await asyncio.sleep(1)
                        except asyncio.CancelledError:
                            logger.debug(f"Server task for '{server_name}' cancelled")
                            raise

                task = asyncio.create_task(server_task(), name=f"mcp_server_{server_name}")

                # Store the server and its task
                self._servers[server_name] = server_instance
                self._server_tasks[server_name] = task

                logger.info(f"Started MCP server: {server_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to start MCP server '{server_name}': {e}")
                return False

    async def stop_server(self, server_name: str) -> bool:
        """
        Stop an individual MCP server.

        Args:
            server_name: Name identifier for the server to stop

        Returns:
            True if stopped successfully, False if not found
        """
        async with self._lock:
            if server_name not in self._servers:
                # Server already stopped - this is fine, return success
                return True

            try:
                # Cancel the server task
                task = self._server_tasks[server_name]
                task.cancel()

                # Wait for task to complete cancellation with timeout
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass  # Expected when we cancel the task or it times out

                # Remove from tracking
                del self._servers[server_name]
                del self._server_tasks[server_name]

                logger.info(f"Stopped MCP server: {server_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to stop MCP server '{server_name}': {e}")
                # Clean up tracking even if stop failed
                if server_name in self._servers:
                    del self._servers[server_name]
                if server_name in self._server_tasks:
                    del self._server_tasks[server_name]
                return False

    async def restart_server(self, server_name: str, server_instance: MCPServerStdio | MCPServerStreamableHTTP) -> bool:
        """
        Restart an individual MCP server.

        Args:
            server_name: Name identifier for the server
            server_instance: The new MCP server instance

        Returns:
            True if restarted successfully, False otherwise
        """
        logger.info(f"Restarting MCP server: {server_name}")
        await self.stop_server(server_name)
        return await self.start_server(server_name, server_instance)

    async def start_enabled_servers(self) -> None:
        """
        Start all enabled MCP servers from database (used at app startup).
        """
        try:
            # Get enabled servers from database with their names
            async with AsyncSessionLocal() as db:
                filters = [MCPServer.enabled]
                db_servers = await crud_mcp_server.filter(db=db, filters=filters)

            if not db_servers:
                logger.info("No enabled MCP servers configured")
                return

            logger.info(f"Starting {len(db_servers)} enabled MCP servers...")

            # Start all servers concurrently
            startup_tasks = []
            for db_server in db_servers:
                task = asyncio.create_task(self._start_single_server(db_server), name=f"start_{db_server.name}")
                startup_tasks.append(task)

            # Execute all startups concurrently
            results = await asyncio.gather(*startup_tasks, return_exceptions=True)

            # Count successes and log failures
            started_count = sum(1 for result in results if result is True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to start server '{db_servers[i].name}': {result}")

            logger.info(f"Successfully started {started_count}/{len(db_servers)} MCP servers")

        except Exception as e:
            logger.error(f"Failed to start MCP servers: {e}")
            raise

    async def _start_single_server(self, db_server) -> bool:
        """Start a single server with error handling for concurrent startup."""
        try:
            server_instance = create_server_instance_from_db(db_server)
            if server_instance:
                return await self.start_server(db_server.name, server_instance)
            return False
        except Exception as e:
            logger.error(f"Failed to start MCP server '{db_server.name}': {e}")
            return False

    async def get_running_servers(self) -> list[MCPServerStdio | MCPServerStreamableHTTP]:
        """
        Get all currently running MCP servers.

        Returns:
            List of running MCP server instances ready for agent use
        """
        return list(self._servers.values())

    async def shutdown(self) -> None:
        """
        Shutdown all running MCP servers (used at app shutdown).
        """
        try:
            async with self._lock:
                if not self._servers:
                    logger.debug("No MCP servers running, nothing to shutdown")
                    return

                logger.info(f"Shutting down {len(self._servers)} MCP servers...")

                # Stop each server individually with timeout
                server_names = list(self._servers.keys())
                for server_name in server_names:
                    try:
                        await asyncio.wait_for(self.stop_server(server_name), timeout=3.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout stopping server '{server_name}', forcing cleanup")
                        # Force cleanup if timeout
                        if server_name in self._servers:
                            del self._servers[server_name]
                        if server_name in self._server_tasks:
                            del self._server_tasks[server_name]
                    except Exception as e:
                        logger.error(f"Error stopping server '{server_name}': {e}")

                logger.info("All MCP servers have been shut down")
        except asyncio.CancelledError:
            # Handle graceful shutdown cancellation
            logger.info("MCP server shutdown was cancelled")
            raise


# Global singleton instance
mcp_lifecycle_manager = MCPServerLifecycleManager()
