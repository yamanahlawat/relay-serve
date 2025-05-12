from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import (
    MCPServerCreate,
    MCPServerResponse,
    MCPServerUpdate,
    ServerStatus,
)
from app.model_context_protocol.schemas.tools import MCPTool
from app.model_context_protocol.services.lifecycle import mcp_lifecycle_manager


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

    async def create_server(self, server_data: MCPServerCreate) -> MCPServerResponse:
        """
        Create a new MCP server configuration.
        This method creates a new server configuration in the database
        and starts the server if it's enabled.
        Args:
            server_data: Server configuration data
        Returns:
            Created server with status
        """
        # Create server in database
        db_server = await crud_mcp_server.create(db=self.db, obj_in=server_data)

        # Determine server status
        status = ServerStatus.STOPPED
        available_tools = []

        # Start server if enabled
        if db_server.enabled:
            try:
                # Create server config
                config = await self.lifecycle_manager.get_server_config(self.db, db_server.name)
                if config:
                    # Start server
                    session = await self.lifecycle_manager.process_manager.start_server(
                        server_name=db_server.name, config=config
                    )
                    status = ServerStatus.RUNNING

                    # Get available tools
                    try:
                        tools_response = await session.list_tools()
                        available_tools = [
                            MCPTool(
                                name=tool.name,
                                description=tool.description,
                                server_name=db_server.name,
                                input_schema=tool.inputSchema,
                            )
                            for tool in tools_response.tools
                        ]
                    except Exception:
                        pass  # Ignore errors when fetching tools
            except Exception:
                status = ServerStatus.ERROR

        # Create response
        return MCPServerResponse(
            **db_server.__dict__,
            status=status,
            available_tools=available_tools,
        )

    async def update_server(self, server_id: UUID, update_data: MCPServerUpdate) -> MCPServerResponse:
        """
        Update an existing MCP server configuration.

        This method updates an existing server configuration in the database
        and manages its runtime state based on the updated configuration.

        Args:
            server_id: UUID of the server to update
            update_data: Server configuration update data

        Returns:
            MCPServerResponse with updated server data and status

        Raises:
            MCPServerError: If the server is not found
        """
        # Get the server from database
        existing = await crud_mcp_server.get(db=self.db, id=server_id)
        if not existing:
            raise MCPServerError("Server not found")

        # Update the server in database
        updated = await crud_mcp_server.update(db=self.db, id=server_id, obj_in=update_data)

        # Get running servers to determine status
        running_servers = await self.lifecycle_manager.get_running_servers()
        status = ServerStatus.UNKNOWN
        available_tools = []

        # Manage server runtime state based on updated configuration
        if updated.enabled:
            # Server should be running
            restart_needed = False

            # Check if server is already running
            if updated.name in running_servers:
                # Server is already running, but config may have changed
                # Restart the server to apply new configuration
                await self.lifecycle_manager.process_manager.shutdown_server(updated.name)
                restart_needed = True
            else:
                # Server is not running, needs to be started
                restart_needed = True

            # Start server if needed
            if restart_needed:
                try:
                    config = await self.lifecycle_manager.get_server_config(self.db, updated.name)
                    if config:
                        session = await self.lifecycle_manager.process_manager.start_server(
                            server_name=updated.name, config=config
                        )
                        status = ServerStatus.RUNNING

                        # Get available tools
                        try:
                            tools_response = await session.list_tools()
                            available_tools = [
                                MCPTool(
                                    name=tool.name,
                                    description=tool.description,
                                    server_name=updated.name,
                                    input_schema=tool.inputSchema,
                                )
                                for tool in tools_response.tools
                            ]
                        except Exception:
                            pass  # Ignore errors when fetching tools
                    else:
                        status = ServerStatus.ERROR
                except Exception:
                    status = ServerStatus.ERROR
        else:
            # Server should be stopped
            if updated.name in running_servers:
                await self.lifecycle_manager.process_manager.shutdown_server(updated.name)
            status = ServerStatus.STOPPED

        # Always return MCPServerResponse
        return MCPServerResponse(
            **updated.__dict__,
            status=status,
            available_tools=available_tools,
        )

    async def delete_server(self, server_id: UUID) -> None:
        """
        Delete an existing MCP server configuration.
        This method deletes an existing server configuration from the database
        and shuts down the server if it's running.
        Args:
            server_id: UUID of the server to delete
        Raises:
            MCPServerError: If the server is not found
        """
        # Get the server from database
        existing = await crud_mcp_server.get(db=self.db, id=server_id)
        if not existing:
            raise MCPServerError("Server not found")

        # Shutdown server if running
        running_servers = await self.lifecycle_manager.get_running_servers()
        if existing.name in running_servers:
            await self.lifecycle_manager.process_manager.shutdown_server(existing.name)

        # Delete the server from database
        await crud_mcp_server.delete(db=self.db, id=server_id)
