from typing import Any, Dict, Optional

from loguru import logger
from mcp import ClientSession
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.exceptions import MCPServerError, MCPServerNotFoundError
from app.model_context_protocol.schemas.servers import MCPServerBase, MCPServerUpdate
from app.model_context_protocol.services.process import MCPProcessManager


class MCPServerLifecycleManager:
    """
    Manages the lifecycle of MCP servers. Responsible for starting, stopping,
    and monitoring server processes based on their configurations.

    Works with MCPProcessManager to handle the actual process execution while
    maintaining the higher-level lifecycle state.
    """

    def __init__(self) -> None:
        self.process_manager = MCPProcessManager()

    def _create_server_config(self, db_server: Any) -> MCPServerBase:
        """
        Create a server configuration object from a database server model.

        Args:
            db_server: Database server model

        Returns:
            MCPServerBase configuration object
        """
        config = db_server.config or {}
        return MCPServerBase(
            command=db_server.command,
            server_type=db_server.server_type,
            config=config,
            enabled=db_server.enabled,
            env=db_server.env,
        )

    async def start_enabled_servers(self) -> None:
        """
        Start all enabled servers from the database.
        """
        async with AsyncSessionLocal() as db:
            # Query all enabled servers from database using CRUD
            enabled_servers = await crud_mcp_server.filter(
                db,
                filters=[crud_mcp_server.model.enabled],
            )

            for server in enabled_servers:
                await self._start_server_if_needed(server)

    async def _start_server_if_needed(self, server: Any) -> None:
        """
        Start a server if it's not already running.

        Args:
            server: Server database model
        """
        # Check if server is already running
        if not await self.process_manager.get_session(server.name):
            try:
                server_config = self._create_server_config(server)
                await self.process_manager.start_server(server_name=server.name, config=server_config)
                logger.info(f"Successfully started MCP server: {server.name}")
            except Exception as error:
                logger.error(f"Failed to start MCP server {server.name}: {error}")

    async def get_server_session(self, db: AsyncSession, name: str) -> ClientSession:
        """
        Get a server session by name. Starts the server if it's not running.
        Args:
            db: Database session
            name: Name of the server to get session for
        Returns:
            ClientSession: Active server session
        Raises:
            MCPServerNotFoundError: If server doesn't exist in database or is disabled
            MCPServerError: If server fails to start
        """
        # Get server from database using CRUD
        db_server = await crud_mcp_server.get_by_name(db, name=name)
        if not db_server:
            raise MCPServerNotFoundError(f"Server '{name}' not found in database")

        if not db_server.enabled:
            raise MCPServerNotFoundError(f"Server '{name}' is disabled")

        # Check if server is already running
        session = await self.process_manager.get_session(name=name)
        if session:
            return session

        # Create config and start server
        server_config = self._create_server_config(db_server)

        try:
            return await self.process_manager.start_server(server_name=name, config=server_config)
        except Exception as error:
            raise MCPServerError(f"Failed to start server '{name}': {error}")

    async def get_server_config(self, db: AsyncSession, name: str) -> Optional[MCPServerBase]:
        """
        Get server configuration by name from the database.
        Args:
            db: Database session
            name: Server name
        Returns:
            Server configuration if found, None otherwise
        """
        db_server = await crud_mcp_server.get_by_name(db, name=name)
        if not db_server:
            return None

        return self._create_server_config(db_server)

    async def update_server_config(self, db: AsyncSession, name: str, config: MCPServerBase) -> None:
        """
        Update a server configuration in the database and restart if necessary.

        Args:
            db: Database session
            name: Server name to update
            config: New server configuration
        """
        # Update in database
        update_data = MCPServerUpdate(
            command=config.command,
            server_type=config.server_type,
            config=config.config,
            enabled=config.enabled,
            env=config.env,
        )

        db_server = await crud_mcp_server.get_by_name(db, name=name)
        if not db_server:
            logger.error(f"Cannot update non-existent server: {name}")
            return

        await crud_mcp_server.update(db, id=db_server.id, obj_in=update_data)
        await self._manage_server_state(name, config)

    async def _manage_server_state(self, name: str, config: MCPServerBase) -> None:
        """
        Manage server runtime state based on configuration.
        Args:
            name: Server name
            config: Server configuration
        """
        session = await self.process_manager.get_session(name)

        if session and not config.enabled:
            # Shutdown server if it's now disabled
            await self.process_manager.shutdown_server(name)
            logger.info(f"Shut down disabled MCP server: {name}")
        elif not session and config.enabled:
            # Start server if it's now enabled
            try:
                await self.process_manager.start_server(server_name=name, config=config)
                logger.info(f"Started updated MCP server: {name}")
            except Exception as error:
                logger.error(f"Failed to start updated MCP server {name}: {error}")

    async def remove_server(self, db: AsyncSession, name: str) -> bool:
        """
        Remove a server from the database and shut it down if running.
        Args:
            db: Database session
            name: Server name to remove
        Returns:
            Whether the server was successfully removed
        """
        # Check if server exists
        db_server = await crud_mcp_server.get_by_name(db, name=name)
        if not db_server:
            return False

        # Shutdown server if running
        session = await self.process_manager.get_session(name)
        if session:
            await self.process_manager.shutdown_server(name)

        # Remove from database
        await crud_mcp_server.delete(db, id=db_server.id)
        return True

    async def get_running_servers(self) -> Dict[str, ClientSession]:
        """
        Get all currently running servers.
        This uses the in-memory state from the process manager rather than the database,
        as it refers to actually running processes.
        """
        return await self.process_manager.list_sessions()

    async def shutdown(self) -> None:
        """
        Shutdown all running servers gracefully.

        This method ensures all servers are properly shut down, even if errors occur.
        It's designed to be called during application shutdown.
        """
        try:
            await self.process_manager.shutdown()
            logger.info("MCP server lifecycle manager completed shutdown")
        except Exception as e:
            logger.error(f"Error during MCP lifecycle manager shutdown: {e}")
            # Ensure we don't propagate exceptions during shutdown
            # as this could prevent the application from shutting down cleanly


# Create a singleton instance of the service
mcp_lifecycle_manager = MCPServerLifecycleManager()
