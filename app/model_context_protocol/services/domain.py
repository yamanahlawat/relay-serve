from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.initialize import mcp_lifecycle_manager
from app.model_context_protocol.schemas.servers import (
    MCPServerResponse,
    MCPServerToggleResponse,
    MCPServerUpdate,
    ServerStatus,
)
from app.model_context_protocol.schemas.tools import MCPTool


class MCPServerDomainService:
    """
    Domain service for MCP server operations.

    This service handles domain operations and business logic for MCP servers,
    focusing on database interactions and API-level operations. It uses the
    MCPServerLifecycleManager for runtime operations.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.lifecycle_manager = mcp_lifecycle_manager

    async def list_servers(self, offset: int = 0, limit: int = 10) -> list[MCPServerResponse]:
        """
        List all configured MCP servers with their status and available tools.
        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
        Returns:
            List of all server responses with status and tools
        """
        # Get all servers from DB
        servers = await crud_mcp_server.filter(
            db=self.db,
            offset=offset,
            limit=limit,
        )

        # Get running servers to determine status
        running_servers = await self.lifecycle_manager.get_running_servers()

        response = []
        for server in servers:
            # Get available tools for running server
            available_tools = []
            status = ServerStatus.STOPPED

            # Check if server is running and get tools if it is
            if server.name in running_servers:
                status = ServerStatus.RUNNING
                session = running_servers[server.name]
                try:
                    tools_response = await session.list_tools()
                    available_tools = [
                        MCPTool(
                            name=tool.name,
                            description=tool.description,
                            server_name=server.name,
                            input_schema=tool.inputSchema,
                        )
                        for tool in tools_response.tools
                    ]
                except Exception:
                    pass  # Ignore errors when fetching tools
            elif not server.enabled:
                status = ServerStatus.DISABLED

            # Create server response object
            server_response = MCPServerResponse(
                **server.__dict__,
                status=status,
                available_tools=available_tools,
            )
            response.append(server_response)

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

        # Get running servers to determine status
        running_servers = await self.lifecycle_manager.get_running_servers()
        status = ServerStatus.UNKNOWN

        # Update server runtime state based on new enabled status
        if updated.enabled:
            # Server should be running
            if updated.name not in running_servers:
                try:
                    # Get server config and start it
                    config = await self.lifecycle_manager.get_server_config(self.db, updated.name)
                    if config:
                        await self.lifecycle_manager.process_manager.start_server(
                            server_name=updated.name, config=config
                        )
                        status = ServerStatus.RUNNING
                    else:
                        status = ServerStatus.ERROR
                except Exception:
                    status = ServerStatus.ERROR
            else:
                status = ServerStatus.RUNNING
        else:
            # Server should be stopped
            if updated.name in running_servers:
                await self.lifecycle_manager.process_manager.shutdown_server(updated.name)
            status = ServerStatus.STOPPED

        return MCPServerToggleResponse(
            name=updated.name,
            enabled=updated.enabled,
            status=status,
        )
