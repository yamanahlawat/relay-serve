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
        self.exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}

    async def start_server(self, server_name: str, config: MCPServerBase) -> ClientSession:
        """
        Start an MCP server process and return its session.
        """
        try:
            return await self._start_command_server(server_name, config)
        except Exception as error:
            logger.exception(f"Failed to start MCP server {server_name}")
            raise MCPServerError(f"Failed to start server: {error}")

    async def _start_command_server(self, server_name: str, config: MCPServerBase) -> ClientSession:
        """
        Start an MCP server using direct command execution and return its session.
        """
        # Create server parameters
        # If env is provided, merge it with the current environment instead of replacing it
        server_env = None
        if config.env:
            # Extract secret values from SecretStr objects and preserve the current environment
            extracted_env = {key: value.get_secret_value() for key, value in config.env.items()}
            server_env = {**os.environ, **extracted_env}

        server_params = StdioServerParameters(command=config.command, args=config.args, env=server_env)

        # Start server process and get transport
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport

        # Create and initialize session
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        # Store session
        self._sessions[server_name] = session

        # List available tools
        response = await session.list_tools()
        logger.info(f"Started MCP server {server_name} with tools: {[tool.name for tool in response.tools]}")

        return session

    async def get_session(self, name: str) -> ClientSession | None:
        """
        Get the session for a server if it exists.
        """
        return self._sessions.get(name)

    async def list_sessions(self) -> dict[str, ClientSession]:
        """
        Get all active server sessions.
        """
        return self._sessions

    async def shutdown_server(self, name: str) -> None:
        """
        Shutdown a specific MCP server process by name.

        Args:
            name: Name of the server to shut down
        """
        session = self._sessions.get(name)
        if not session:
            logger.warning(f"Attempted to shut down non-running server: {name}")
            return

        try:
            # The session is managed by the exit_stack, so we need to remove it
            # from our sessions dict before it's closed by the exit_stack
            self._sessions.pop(name, None)

            # Since we registered the session with exit_stack in _start_command_server,
            # we can't directly close it here. Instead, we need to rely on the exit_stack
            # to close it when the application shuts down.
            #
            # If we need immediate cleanup, we would need to track the exit_stack context
            # for each session and close it individually.

            logger.info(f"Removed MCP server {name} from active sessions")
        except Exception as e:
            logger.error(f"Error shutting down MCP server {name}: {e}")
            # Still remove from sessions dict even if there's an error
            self._sessions.pop(name, None)

    async def shutdown(self) -> None:
        """
        Shutdown all MCP server processes.
        """
        # Close all server sessions
        await self.exit_stack.aclose()
        self._sessions.clear()

        logger.info("Shut down all MCP servers")
