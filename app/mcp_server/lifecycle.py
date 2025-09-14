"""
MCP Server Lifecycle Management Service.

This service manages the lifecycle of MCP servers at the application level,
ensuring servers are started once during app startup and reused across
all chat sessions and agent interactions.
"""

import asyncio
from contextlib import AsyncExitStack

from loguru import logger
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

from app.core.database.session import AsyncSessionLocal
from app.mcp_server.crud import crud_mcp_server
from app.mcp_server.model import MCPServer
from app.mcp_server.utils import create_server_instance_from_db

# Timeout constants for consistent behavior
SERVER_STOP_TIMEOUT = 10.0
APPLICATION_SHUTDOWN_TIMEOUT = 20.0
REMAINING_TASKS_TIMEOUT = 3.0
SERVER_INIT_TIMEOUT = 30.0


class MCPServerLifecycleManager:
    """
    Manages the lifecycle of individual MCP servers using dedicated tasks and AsyncExitStack.

    This singleton service:
    - Manages servers individually for granular control
    - Uses individual lifecycle tasks per server to prevent task crossing issues
    - Each task owns its own AsyncExitStack for proper resource management
    - Handles graceful startup and shutdown following MCP 2025 best practices
    """

    def __init__(self) -> None:
        """Initialize the lifecycle manager."""
        self._servers: dict[str, MCPServerStdio | MCPServerStreamableHTTP] = {}
        self._lifecycle_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_events: dict[str, asyncio.Event] = {}
        self._init_events: dict[str, asyncio.Event] = {}
        self._init_results: dict[str, bool] = {}
        self._lock = asyncio.Lock()
        self._is_shutting_down = False

    async def start_server(self, server_name: str, server_instance: MCPServerStdio | MCPServerStreamableHTTP) -> bool:
        """
        Start an individual MCP server using dedicated lifecycle task.

        Args:
            server_name: Name identifier for the server
            server_instance: The MCP server instance to start

        Returns:
            True if started successfully, False otherwise
        """
        # Check if server already exists and stop it first (outside the lock to avoid deadlock)
        should_stop_existing = False
        async with self._lock:
            if server_name in self._lifecycle_tasks:
                should_stop_existing = True

        if should_stop_existing:
            logger.info(f"Server '{server_name}' already exists, restarting it")
            await self.stop_server(server_name)

        # Setup phase - acquire lock only for state modification
        init_event = asyncio.Event()
        shutdown_event = asyncio.Event()

        async with self._lock:
            if self._is_shutting_down:
                logger.warning(f"Cannot start server '{server_name}' - manager is shutting down")
                return False

            # Create initialization coordination events
            self._init_events[server_name] = init_event
            self._init_results[server_name] = False

            # Create shutdown signal event
            self._shutdown_events[server_name] = shutdown_event

            # Start a dedicated task to manage the server's lifecycle
            server_task = asyncio.create_task(
                self._server_lifecycle_task(server_name, server_instance, init_event, shutdown_event),
                name=f"mcp_server_{server_name}",
            )
            self._lifecycle_tasks[server_name] = server_task

        # Wait for initialization outside the lock to avoid blocking other operations
        try:
            await asyncio.wait_for(init_event.wait(), timeout=SERVER_INIT_TIMEOUT)

            # Check if initialization was successful
            success = self._init_results.get(server_name, False)
            if success:
                logger.info(f"Started MCP server: {server_name}")
                return True
            else:
                logger.error(f"Failed to initialize MCP server: {server_name}")
                await self._cleanup_server_resources(server_name)
                return False

        except asyncio.TimeoutError:
            logger.error(f"Timeout starting MCP server '{server_name}'")
            await self._cleanup_server_resources(server_name)
            return False
        except Exception as e:
            logger.error(f"Failed to start MCP server '{server_name}': {e}")
            await self._cleanup_server_resources(server_name)
            return False

    async def _server_lifecycle_task(
        self,
        server_name: str,
        server_instance: MCPServerStdio | MCPServerStreamableHTTP,
        init_event: asyncio.Event,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Task that manages the entire lifecycle of a server within a single task context.
        This prevents task crossing issues by ensuring the task that creates the context also disposes of it.
        """
        # This task exclusively owns the AsyncExitStack for this server
        async with AsyncExitStack() as server_context:
            try:
                # Enter the server instance into the context
                await server_context.enter_async_context(server_instance)

                # Store server for access
                async with self._lock:
                    self._servers[server_name] = server_instance

                # Signal initialization success
                self._init_results[server_name] = True
                init_event.set()

                logger.info(f"MCP server {server_name} lifecycle task started successfully")

                # Wait for shutdown signal
                await shutdown_event.wait()
                logger.info(f"Server {server_name} lifecycle task completing, context will be properly closed")

            except Exception as e:
                # Signal initialization failure if init hasn't been set yet
                logger.exception(f"Error in server lifecycle for {server_name}: {e}")
                self._init_results[server_name] = False
                if not init_event.is_set():
                    init_event.set()  # Signal that initialization is done (but failed)

            finally:
                # Remove from servers dict
                async with self._lock:
                    self._servers.pop(server_name, None)

    async def _cleanup_server_resources(self, server_name: str) -> None:
        """Clean up all resources associated with a server."""
        async with self._lock:
            self._lifecycle_tasks.pop(server_name, None)
            self._servers.pop(server_name, None)
            self._shutdown_events.pop(server_name, None)
            self._init_events.pop(server_name, None)
            self._init_results.pop(server_name, None)

    async def stop_server(self, server_name: str) -> bool:
        """
        Stop an individual MCP server by signaling its lifecycle task.

        Args:
            server_name: Name identifier for the server to stop

        Returns:
            True if stopped successfully, False if not found
        """
        # Get shutdown event while holding lock to prevent race conditions
        shutdown_event = None
        lifecycle_task = None

        async with self._lock:
            shutdown_event = self._shutdown_events.get(server_name)
            lifecycle_task = self._lifecycle_tasks.get(server_name)

            if not shutdown_event and not lifecycle_task:
                return True

        try:
            # Signal the lifecycle task to exit, which will properly close the context
            if shutdown_event:
                shutdown_event.set()

            # Wait for the lifecycle task to complete
            if lifecycle_task:
                try:
                    await asyncio.wait_for(lifecycle_task, timeout=SERVER_STOP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"Timed out waiting for lifecycle task of {server_name} to complete")
                except asyncio.CancelledError:
                    logger.warning(f"Server shutdown for {server_name} was cancelled")

            await self._cleanup_server_resources(server_name)
            logger.info(f"Stopped MCP server: {server_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop MCP server '{server_name}': {e}")
            # Clean up tracking even if stop failed
            await self._cleanup_server_resources(server_name)
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
        async with self._lock:
            return list(self._servers.values())

    async def get_server_names(self) -> list[str]:
        """Get names of all currently running servers."""
        async with self._lock:
            return list(self._servers.keys())

    async def is_server_running(self, server_name: str) -> bool:
        """Check if a specific server is running."""
        async with self._lock:
            return server_name in self._servers

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all running MCP servers by signaling their lifecycle tasks.
        """
        async with self._lock:
            self._is_shutting_down = True
            server_names = list(self._servers.keys())

        if not server_names:
            logger.debug("No MCP servers running, nothing to shutdown")
            return

        logger.info(f"Shutting down {len(server_names)} MCP servers...")

        try:
            # Signal all servers to shutdown concurrently
            shutdown_tasks = []
            for server_name in server_names:
                shutdown_task = asyncio.create_task(self.stop_server(server_name), name=f"shutdown_{server_name}")
                shutdown_tasks.append(shutdown_task)

            try:
                # Wait for all servers to shutdown with timeout
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True), timeout=APPLICATION_SHUTDOWN_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout during graceful shutdown, forcing cleanup")
            except Exception as e:
                logger.error(f"Error during graceful shutdown: {e}")

            # Force cleanup any remaining tracking
            async with self._lock:
                remaining_tasks = list(self._lifecycle_tasks.values())

            if remaining_tasks:
                logger.info(f"Waiting for {len(remaining_tasks)} remaining lifecycle tasks")
                try:
                    _, pending = await asyncio.wait(remaining_tasks, timeout=REMAINING_TASKS_TIMEOUT)
                    if pending:
                        logger.warning(f"{len(pending)} lifecycle tasks did not complete in time")
                except Exception as e:
                    logger.error(f"Error waiting for remaining tasks: {e}")

            # Clear all tracking
            async with self._lock:
                self._servers.clear()
                self._lifecycle_tasks.clear()
                self._shutdown_events.clear()
                self._init_events.clear()
                self._init_results.clear()
                self._is_shutting_down = False

            logger.info("All MCP servers have been shut down")
        except asyncio.CancelledError:
            logger.info("MCP server shutdown was cancelled")
            raise


# Global singleton instance
mcp_lifecycle_manager = MCPServerLifecycleManager()
