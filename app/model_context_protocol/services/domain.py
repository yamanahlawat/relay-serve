from uuid import UUID

from loguru import logger
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP
from sqlalchemy.ext.asyncio import AsyncSession

from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import (
    MCPServerBase,
    MCPServerCreate,
    MCPServerResponse,
    MCPServerUpdate,
    ServerStatus,
)
from app.model_context_protocol.services.lifecycle import mcp_lifecycle_manager
from app.model_context_protocol.services.validator import MCPServerValidator
from app.model_context_protocol.utils import create_server_instance_from_db


class MCPServerDomainService:
    """
    Domain service for MCP server operations.

    This service handles domain operations and business logic for MCP servers,
    focusing on database interactions and API-level operations.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.validator = MCPServerValidator()

    async def list_servers(self, offset: int = 0, limit: int = 10) -> list[MCPServerResponse]:
        """
        List all configured MCP servers with their status.
        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
        Returns:
            List of all server responses with status (no tools - lazy loaded)
        """
        # Get all servers from DB
        servers = await crud_mcp_server.filter(
            db=self.db,
            offset=offset,
            limit=limit,
        )

        response = []
        for server in servers:
            # Simple status determination based on configuration
            status = ServerStatus.DISABLED if not server.enabled else ServerStatus.RUNNING

            # Create server response object without tools (lazy loaded)
            server_response = MCPServerResponse(
                **server.__dict__,
                status=status,
            )
            response.append(server_response)

        return response

    async def create_server(self, server_data: MCPServerCreate) -> MCPServerResponse:
        """
        Create a new MCP server configuration.
        This method creates a new server configuration in the database and validates it.
        If validation fails, the server creation is rejected.
        Args:
            server_data: Server configuration data
        Returns:
            Created server with status (no tools - lazy loaded)
        Raises:
            MCPServerError: If server validation fails
        """
        # Validate server configuration before creating
        if server_data.enabled:
            # Convert to schema for validation
            server_config = MCPServerBase(
                command=server_data.command,
                server_type=server_data.server_type,
                config=server_data.config,
                enabled=server_data.enabled,
                env=server_data.env,
            )

            # Validate server configuration
            is_valid, error_msg = await self.validator.validate_server(server_data.name, server_config)

            if not is_valid:
                raise MCPServerError(f"Server validation failed: {error_msg}")

            logger.info(f"Successfully validated server '{server_data.name}'")

        # Create server in database (only if validation passed)
        db_server = await crud_mcp_server.create(db=self.db, obj_in=server_data)

        # Determine status based on enabled flag
        status = ServerStatus.DISABLED if not db_server.enabled else ServerStatus.RUNNING

        # Create response without tools (lazy loaded)
        return MCPServerResponse(
            **db_server.__dict__,
            status=status,
        )

    async def update_server(self, server_id: UUID, update_data: MCPServerUpdate) -> MCPServerResponse:
        """
        Update an existing MCP server configuration.

        This method updates an existing server configuration in the database and validates it.
        If validation fails, the server update is rejected.

        Args:
            server_id: UUID of the server to update
            update_data: Server configuration update data

        Returns:
            MCPServerResponse with updated server data and status (no tools - lazy loaded)

        Raises:
            MCPServerError: If the server is not found or validation fails
        """
        # Get the server from database
        existing = await crud_mcp_server.get(db=self.db, id=server_id)
        if not existing:
            raise MCPServerError("Server not found")

        # Update the server in database
        updated = await crud_mcp_server.update(db=self.db, id=server_id, obj_in=update_data)

        if not updated:
            raise MCPServerError(f"Server with ID {server_id} not found")

        # Validate updated server configuration if enabled
        if updated.enabled:
            # Convert database model to schema for validation
            server_config = MCPServerBase(
                command=updated.command,
                server_type=updated.server_type,
                config=updated.config,
                enabled=updated.enabled,
                env=updated.env,
            )

            # Validate server configuration
            is_valid, error_msg = await self.validator.validate_server(updated.name, server_config)

            if not is_valid:
                # Revert the update by setting enabled=False
                revert_data = MCPServerUpdate(enabled=False)
                await crud_mcp_server.update(db=self.db, id=server_id, obj_in=revert_data)
                raise MCPServerError(f"Server validation failed: {error_msg}")

        # Determine status based on enabled flag
        status = ServerStatus.DISABLED if not updated.enabled else ServerStatus.RUNNING

        # Handle individual server status changes efficiently
        await self._handle_server_lifecycle_change(updated_server=updated)

        # Return MCPServerResponse without tools (lazy loaded)
        return MCPServerResponse(
            **updated.__dict__,
            status=status,
        )

    async def delete_server(self, server_id: UUID) -> None:
        """
        Delete an existing MCP server configuration.
        This method deletes an existing server configuration from the database.
        Args:
            server_id: UUID of the server to delete
        Raises:
            MCPServerError: If the server is not found
        """
        # Get the server from database
        existing = await crud_mcp_server.get(db=self.db, id=server_id)
        if not existing:
            raise MCPServerError("Server not found")

        # Delete the server from database
        await crud_mcp_server.delete(db=self.db, id=server_id)

    async def get_running_servers_for_agent(self) -> list[MCPServerStdio | MCPServerStreamableHTTP]:
        """
        Get pre-started MCP servers from the lifecycle manager for agent use.

        Returns:
            List of running MCP server instances ready for agent use
        """
        return await mcp_lifecycle_manager.get_running_servers()

    async def _handle_server_lifecycle_change(self, updated_server):
        """Handle individual server lifecycle changes efficiently."""

        if updated_server.enabled:
            # Server is enabled - start or restart it
            server_instance = create_server_instance_from_db(db_server=updated_server)
            if server_instance:
                await mcp_lifecycle_manager.restart_server(
                    server_name=updated_server.name, server_instance=server_instance
                )
                logger.info(f"Started/restarted MCP server: {updated_server.name}")
        else:
            # Server is disabled - stop it
            await mcp_lifecycle_manager.stop_server(server_name=updated_server.name)
            logger.info(f"Stopped MCP server: {updated_server.name}")
