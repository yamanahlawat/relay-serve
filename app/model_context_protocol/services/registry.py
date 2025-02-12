from loguru import logger
from mcp import ClientSession

from app.model_context_protocol.exceptions import MCPServerNotFoundError
from app.model_context_protocol.services.manager import MCPServerManager
from app.model_context_protocol.schemas.servers import MCPConfig, MCPServerConfig


class MCPServerRegistry:
    """
    Registry for MCP servers. Manages server configurations and their operational status.
    Works with MCPServerManager to handle server lifecycle.
    """

    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self.manager = MCPServerManager()

    async def start_enabled_servers(self) -> None:
        """
        Start all enabled servers from the configuration.
        """
        for name, server_config in self.config.enabled_servers.items():
            # Check if server is already running
            if not await self.manager.get_session(name):
                try:
                    await self.manager.start_server(server_name=name, config=server_config)
                    logger.info(f"Successfully started MCP server: {name}")
                except Exception as error:
                    logger.error(f"Failed to start MCP server {name}: {error}")

    async def get_server_session(self, name: str) -> ClientSession:
        """
        Get a server session by name. Starts the server if it's not running.
        Args:
            name: Name of the server to get session for
        Returns:
            ClientSession: Active server session
        Raises:
            MCPServerNotFoundError: If server doesn't exist in configuration
            MCPServerError: If server fails to start
        """
        server_config = self.config.servers.get(name)
        if not server_config:
            raise MCPServerNotFoundError(f"Server '{name}' not found in configuration")

        if not server_config.enabled:
            raise MCPServerNotFoundError(f"Server '{name}' is disabled")

        # Check if server is already running
        session = await self.manager.get_session(name=name)
        if session:
            return session

        # Start server if not running
        return await self.manager.start_server(server_name=name, config=server_config)

    def get_server_config(self, name: str) -> MCPServerConfig | None:
        """
        Get server configuration by name.
        """
        return self.config.servers.get(name)

    async def get_running_servers(self) -> dict[str, ClientSession]:
        """
        Get all currently running servers.
        """
        return await self.manager.list_sessions()

    async def stop_server(self, name: str) -> None:
        """
        Stop a specific server.
        """
        await self.manager.stop_server(name=name)

    async def shutdown(self) -> None:
        """
        Shutdown all running servers.
        """
        await self.manager.shutdown()
