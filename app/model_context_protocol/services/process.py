import asyncio
import os
from contextlib import AsyncExitStack

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import MCPServerBase


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
        # Individual context stacks for each server
        self._contexts: dict[str, AsyncExitStack] = {}
        # Active server sessions
        self._sessions: dict[str, ClientSession] = {}

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
            return await self._start_command_server(server_name, config)
        except Exception as error:
            logger.exception(f"Failed to start MCP server {server_name}")
            raise MCPServerError(f"Failed to start server: {error}")

    async def _start_command_server(self, server_name: str, config: MCPServerBase) -> ClientSession:
        """
        Start an MCP server using direct command execution and return its session.

        Args:
            server_name: Name to identify the server
            config: Server configuration

        Returns:
            ClientSession for the started server
        """
        # If a server with this name already exists, shut it down first
        if server_name in self._sessions:
            await self.shutdown_server(server_name)

        # Create a dedicated context stack for this server
        server_context = AsyncExitStack()
        self._contexts[server_name] = server_context

        try:
            # Create server parameters
            # If env is provided, merge it with the current environment instead of replacing it
            server_env = None
            if config.env:
                # Extract secret values from SecretStr objects and preserve the current environment
                extracted_env = {key: value.get_secret_value() for key, value in config.env.items()}
                server_env = {**os.environ, **extracted_env}

            server_params = StdioServerParameters(command=config.command, args=config.args, env=server_env)

            # Start server process and get transport using the server's context stack
            stdio_transport = await server_context.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport

            # Create and initialize session
            session = await server_context.enter_async_context(ClientSession(stdio, write))
            await session.initialize()

            # Store session
            self._sessions[server_name] = session

            # List available tools
            response = await session.list_tools()
            logger.info(f"Started MCP server {server_name} with tools: {[tool.name for tool in response.tools]}")

            return session
        except Exception as e:
            # Clean up the context if there's an error
            await self._cleanup_server_context(server_name)
            raise e

    async def _cleanup_server_context(self, name: str) -> None:
        """
        Clean up a server's context stack.

        Args:
            name: Name of the server to clean up
        """
        context = self._contexts.pop(name, None)
        if context:
            try:
                await context.aclose()
            except asyncio.CancelledError:
                # Handle asyncio.CancelledError gracefully
                # This can happen during application shutdown or when a task is cancelled
                logger.warning(f"Context cleanup for server {name} was cancelled")
            except Exception as e:
                # Log but don't propagate other exceptions during cleanup
                logger.error(f"Error closing context for server {name}: {e}")
                # Continue with cleanup despite errors

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

    async def shutdown_server(self, name: str) -> None:
        """
        Shutdown a specific MCP server process by name with proper resource cleanup.

        Args:
            name: Name of the server to shut down
        """
        # First remove from sessions dict to prevent further usage
        session = self._sessions.pop(name, None)

        if not session:
            # If the server isn't in our sessions, just clean up any context that might exist
            logger.warning(f"Attempted to shut down non-running server: {name}")
            await self._cleanup_server_context(name)
            return

        try:
            # Clean up the context stack for this server
            await self._cleanup_server_context(name)
            logger.info(f"Shut down MCP server: {name}")
        except asyncio.CancelledError:
            logger.warning(f"Server shutdown for {name} was cancelled")
        except Exception as e:
            logger.error(f"Error shutting down MCP server {name}: {e}")
            # The session has already been removed from the dictionary

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

        # Ensure all contexts are cleaned up, even if shutdown_server failed
        remaining_contexts = list(self._contexts.keys())
        for name in remaining_contexts:
            try:
                await self._cleanup_server_context(name)
            except Exception as e:
                logger.error(f"Error during final context cleanup for {name}: {e}")

        # Clear any remaining sessions and contexts
        self._sessions.clear()
        self._contexts.clear()

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
