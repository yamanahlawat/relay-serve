from contextlib import AsyncExitStack
from typing import Optional

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import MCPServerConfig


class MCPServerManager:
    """
    Manages MCP server lifecycle - starting, stopping, and health checks.
    """

    def __init__(self) -> None:
        self.exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}

    async def start_server(self, server_name: str, config: MCPServerConfig) -> ClientSession:
        """
        Start an MCP server and return its session.
        """
        try:
            # Create server parameters
            server_params = StdioServerParameters(command=config.command, args=config.args, env=config.env)

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

        except Exception as error:
            logger.exception(f"Failed to start MCP server {server_name}")
            raise MCPServerError(f"Failed to start server: {error}")

    async def get_session(self, name: str) -> Optional[ClientSession]:
        """
        Get the session for a server if it exists.
        """
        return self._sessions.get(name)

    async def list_sessions(self) -> dict[str, ClientSession]:
        """
        Get all active server sessions.
        """
        return self._sessions

    async def shutdown(self) -> None:
        """
        Shutdown all MCP servers.
        """
        await self.exit_stack.aclose()
        self._sessions.clear()
        logger.info("Shut down all MCP servers")
