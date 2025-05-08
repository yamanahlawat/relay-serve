from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.initialize import mcp_registry
from app.model_context_protocol.schemas.servers import (
    MCPServerBase,
    MCPServerResponse,
    MCPServerToggleResponse,
    MCPServerUpdate,
    ServerStatus,
)


class MCPServerService:
    """
    Service for managing MCP server configurations.

    In the JSON-driven approach, servers are configured in the DEFAULT_MCP_SERVERS dictionary
    in initialize.py. This service provides methods to view server status and toggle enabled state.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_servers(self, offset: int = 0, limit: int = 10) -> list[MCPServerResponse]:
        """
        List all MCP servers with their status and available tools.
        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
        Returns:
            List of server responses with status and tools
        """
        # Get all servers from DB
        servers = await crud_mcp_server.filter(
            db=self.db,
            offset=offset,
            limit=limit,
        )

        # Get running servers to determine status
        running_servers = await mcp_registry.get_running_servers()

        response = []
        for server in servers:
            status = ServerStatus.RUNNING if server.name in running_servers else ServerStatus.STOPPED

            # Get available tools if server is running
            available_tools = []
            if server.name in running_servers:
                session = running_servers[server.name]
                try:
                    tools_response = await session.list_tools()
                    available_tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        }
                        for tool in tools_response.tools
                    ]
                except Exception:
                    pass  # Ignore errors when fetching tools

            response.append(
                MCPServerResponse(
                    **server.__dict__,
                    status=status,
                    available_tools=available_tools,
                )
            )

        return response

    async def toggle_server(self, server_id: UUID) -> MCPServerToggleResponse:
        """
        Toggle a server's enabled status.

        This is the only runtime modification supported for MCP servers.
        All other configuration changes should be made in the DEFAULT_MCP_SERVERS
        dictionary in initialize.py.

        Args:
            server_id: UUID of the server to toggle

        Returns:
            MCPServerToggleResponse with updated status

        Raises:
            MCPServerError: If the server is not found
        """
        # Get the server from database
        existing = await crud_mcp_server.get(db=self.db, id=server_id)
        if not existing:
            raise MCPServerError("Server not found")

        # Create update object with toggled enabled status
        update_data = MCPServerUpdate(enabled=not existing.enabled)

        # Update the server in database
        updated = await crud_mcp_server.update(db=self.db, id=server_id, obj_in=update_data)

        # Create config for server operations
        config = MCPServerBase(
            command=updated.command,
            args=updated.args,
            enabled=updated.enabled,
            env=updated.env,
        )

        # Start or stop the server based on new enabled status
        running_servers = await mcp_registry.get_running_servers()
        status = ServerStatus.UNKNOWN

        if updated.enabled:
            # Start server if not running
            if updated.name not in running_servers:
                try:
                    await mcp_registry.manager.start_server(server_name=updated.name, config=config)
                    status = ServerStatus.RUNNING
                except Exception:
                    status = ServerStatus.ERROR
            else:
                status = ServerStatus.RUNNING
        else:
            # Stop server if running
            if updated.name in running_servers:
                await mcp_registry.manager.shutdown_server(updated.name)
            status = ServerStatus.STOPPED

        return MCPServerToggleResponse(
            name=updated.name,
            enabled=updated.enabled,
            status=status,
        )
