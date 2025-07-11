import asyncio
import os
from contextlib import AsyncExitStack

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.model_context_protocol.constants import MCPEventType
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import MCPServerBase
from app.model_context_protocol.services.event_bus import mcp_event_bus


class MCPProcessManager:
    """
    Manages MCP server processes - starting, stopping, and monitoring.

    This service is focused purely on process execution and management,
    with no knowledge of domain concepts beyond what's needed for execution.
    It handles the low-level details of process management.
    """

    def __init__(self) -> None:
        # Main exit stack for application-level cleanup
        self.exit_stack = AsyncExitStack()
        # Active server sessions
        self._sessions: dict[str, ClientSession] = {}
        # Tasks that own the server lifecycle - these tasks are responsible for cleanup
        self._lifecycle_tasks: dict[str, asyncio.Task] = {}
        # Events to signal server shutdown
        self._shutdown_events: dict[str, asyncio.Event] = {}
        # Initialization events to coordinate server startup
        self._init_events: dict[str, asyncio.Event] = {}
        # Initialization results for returning session
        self._init_results: dict[str, ClientSession | None] = {}

    async def start_server(self, server_name: str, config: MCPServerBase) -> ClientSession:
        """
        Start an MCP server process and return its session.

        Args:
            server_name: Name to identify the server
            config: Server configuration

        Returns:
            ClientSession for the started server

        Raises:
            MCPServerError: If server fails to start
        """
        try:
            # If a server with this name already exists, shut it down first
            if server_name in self._lifecycle_tasks:
                await self.shutdown_server(server_name)

            # Create initialization coordination events
            init_event = asyncio.Event()
            self._init_events[server_name] = init_event
            self._init_results[server_name] = None

            # Create shutdown signal event
            shutdown_event = asyncio.Event()
            self._shutdown_events[server_name] = shutdown_event

            # Start a dedicated task to manage the server's lifecycle
            # This task will own the AsyncExitStack and handle proper cleanup
            server_task = asyncio.create_task(
                self._server_lifecycle_task(server_name, config, init_event, shutdown_event)
            )
            self._lifecycle_tasks[server_name] = server_task

            # Wait for initialization to complete or fail
            await init_event.wait()

            # Check if initialization was successful
            session = self._init_results.get(server_name)
            if not session:
                raise MCPServerError(f"Failed to initialize server {server_name}")

            # Store session for convenience access
            self._sessions[server_name] = session
            return session
        except Exception as error:
            logger.exception(f"Failed to start MCP server {server_name}")
            # Clean up any partial initialization
            try:
                if server_name in self._lifecycle_tasks:
                    await self.shutdown_server(server_name)
            except Exception:
                pass  # Ignore cleanup errors during startup failure
            raise MCPServerError(f"Failed to start server: {error}")

    async def _server_lifecycle_task(
        self, server_name: str, config: MCPServerBase, init_event: asyncio.Event, shutdown_event: asyncio.Event
    ) -> None:
        """
        Task that manages the entire lifecycle of a server within a single task context.
        This prevents task crossing issues by ensuring the task that creates the context also disposes of it.

        Args:
            server_name: Name to identify the server
            config: Server configuration
            init_event: Event to signal when initialization is complete
            shutdown_event: Event to wait for shutdown signal
        """
        # This task exclusively owns the AsyncExitStack for this server
        # and will be the only one to enter and exit it, preventing task crossing issues
        async with AsyncExitStack() as server_context:
            try:
                # Prepare environment and start server
                server_env = self._prepare_server_environment(config)
                command_args = config.config.get("args", []) if config.config else []
                server_params = StdioServerParameters(command=config.command, args=command_args, env=server_env)

                # Start server process and get transport within this context
                stdio_transport = await server_context.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport

                # Create and initialize session
                session = await server_context.enter_async_context(ClientSession(stdio, write))
                await session.initialize()

                # List available tools
                response = await session.list_tools()
                tool_names = [tool.name for tool in response.tools]
                logger.info(f"Started MCP server {server_name} with tools: {tool_names}")

                # Set session in results and signal initialization success
                self._init_results[server_name] = session
                init_event.set()

                # Publish server started event
                await self._publish_server_event(MCPEventType.SERVER_STARTED, server_name, tools=tool_names)

                # Wait for shutdown signal
                await shutdown_event.wait()
                logger.info(f"Server {server_name} lifecycle task completing, context will be properly closed")

            except Exception as e:
                # Signal initialization failure but don't propagate
                logger.exception(f"Error in server lifecycle for {server_name}: {e}")
                self._init_results[server_name] = None
                init_event.set()  # Signal that initialization is done (but failed)

                # Publish error event
                await self._publish_server_event(MCPEventType.SERVER_ERROR, server_name, error=str(e))

            finally:
                # Clean up tracking dictionaries
                self._cleanup_server_resources(server_name)

    async def get_session(self, name: str) -> ClientSession | None:
        """
        Get the session for a server if it exists.

        Args:
            name: Name of the server

        Returns:
            ClientSession if found, None otherwise
        """
        return self._sessions.get(name)

    async def list_sessions(self) -> dict[str, ClientSession]:
        """
        Get all active server sessions.

        Returns:
            Dictionary of server names to sessions
        """
        return self._sessions

    def _cleanup_server_resources(self, name: str) -> None:
        """
        Remove all resources associated with a server.

        This centralizes the cleanup of all tracking dictionaries to ensure
        consistent resource management and prevent resource leaks.

        Args:
            name: Name of the server to clean up resources for
        """
        self._lifecycle_tasks.pop(name, None)
        self._sessions.pop(name, None)
        self._shutdown_events.pop(name, None)
        self._init_events.pop(name, None)
        self._init_results.pop(name, None)

    def _prepare_server_environment(self, config: MCPServerBase) -> dict[str, str] | None:
        """
        Prepare environment variables for a server process.

        Extracts secret values from SecretStr fields and merges with current environment.

        Args:
            config: Server configuration with environment variables

        Returns:
            Environment dictionary for the server process, or None if no env vars provided
        """
        if not config.env:
            return None

        # Extract secret values and preserve current environment
        extracted_env = {key: value.get_secret_value() for key, value in config.env.items()}
        return {**os.environ, **extracted_env}

    async def _publish_server_event(self, event_type: str, server_name: str, **extra_data) -> None:
        """
        Publish a server event to the event bus with standard format.

        Args:
            event_type: Type of event to publish
            server_name: Name of the server
            **extra_data: Additional data to include in the event
        """
        event_data = {"server_name": server_name, **extra_data}
        await mcp_event_bus.publish(event_type, event_data)

    async def shutdown_server(self, name: str) -> None:
        """
        Shutdown a running MCP server and clean up resources.

        Args:
            name: Name of the server to shut down
        """
        shutdown_event = self._shutdown_events.get(name)
        if not shutdown_event:
            logger.warning(f"Cannot shut down server {name}, no running instance found")
            # No valid server found to shut down, but we can still publish a shutdown event
            # so subscribers (like the tool service) can clean up
            await mcp_event_bus.publish(MCPEventType.SERVER_SHUTDOWN, {"server_name": name})
            return

        try:
            # Signal the lifecycle task to exit, which will properly close the context
            shutdown_event.set()

            # Get the lifecycle task
            lifecycle_task = self._lifecycle_tasks.get(name)

            if lifecycle_task:
                # Wait for the lifecycle task to complete
                try:
                    # Wait with a timeout just to be safe
                    await asyncio.wait_for(lifecycle_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timed out waiting for lifecycle task of {name} to complete")
                except asyncio.CancelledError:
                    logger.warning(f"Server shutdown for {name} was cancelled")

            # Clean up all server resources
            self._cleanup_server_resources(name)

            # Publish server shutdown event
            await mcp_event_bus.publish(MCPEventType.SERVER_SHUTDOWN, {"server_name": name})
            logger.info(f"Shut down MCP server: {name}")
        except Exception as e:
            logger.error(f"Error during server shutdown of {name}: {e}")
            # Still try to publish the shutdown event even on error
            try:
                await mcp_event_bus.publish(MCPEventType.SERVER_SHUTDOWN, {"server_name": name, "error": str(e)})
            except Exception as pub_error:
                logger.error(f"Failed to publish shutdown event for server {name}: {pub_error}")

    async def shutdown(self) -> None:
        """
        Shutdown all MCP server processes gracefully.
        """
        # Make a copy of keys to avoid modification during iteration
        server_names = list(self._sessions.keys())
        shutdown_errors = []

        # Shutdown each server individually
        for name in server_names:
            try:
                await self.shutdown_server(name)
            except Exception as e:
                logger.error(f"Error during shutdown of {name}: {e}")
                shutdown_errors.append((name, str(e)))

        # Wait for any remaining tasks with a timeout
        remaining_tasks = list(self._lifecycle_tasks.values())
        if remaining_tasks:
            try:
                logger.info(f"Waiting for {len(remaining_tasks)} remaining server tasks to complete")
                done, pending = await asyncio.wait(remaining_tasks, timeout=3.0)
                if pending:
                    logger.warning(f"{len(pending)} server tasks did not complete in time")
            except Exception as e:
                logger.error(f"Error waiting for remaining tasks: {e}")

        # Clear any remaining tracking dictionaries
        self._sessions.clear()
        self._lifecycle_tasks.clear()
        self._shutdown_events.clear()
        self._init_events.clear()
        self._init_results.clear()

        try:
            # Close the main exit stack as a fallback
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            logger.warning("Main exit stack cleanup was cancelled")
        except Exception as e:
            logger.error(f"Error during main exit stack cleanup: {e}")

        if shutdown_errors:
            logger.warning(f"Completed MCP server shutdown with {len(shutdown_errors)} errors")
        else:
            logger.info("Shut down all MCP servers successfully")
