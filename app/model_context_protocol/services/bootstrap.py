"""
MCP Server Bootstrap Service

This service is responsible for bootstrapping MCP servers from the database
or seeding default configurations if the database is empty.
"""

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.initialize import MCP_SERVERS, mcp_lifecycle_manager
from app.model_context_protocol.schemas.servers import MCPServerCreate, MCPServerUpdate


class MCPBootstrapService:
    """
    Service for bootstrapping MCP servers.

    Responsible for initializing the MCP system, including seeding default
    configurations and starting enabled servers during application startup.
    """

    def __init__(self) -> None:
        self.lifecycle_manager = mcp_lifecycle_manager

    async def bootstrap(self) -> None:
        """
        Bootstrap MCP servers from database or seed defaults.

        This method:
        1. Checks if there are any server configurations in the database
        2. Seeds default configurations if the database is empty
        3. Starts enabled servers
        """
        async with AsyncSessionLocal() as db:
            await self._do_bootstrap(db)

    async def _do_bootstrap(self, db: AsyncSession) -> None:
        """
        Internal method to perform the actual bootstrapping with a database session.

        This method synchronizes the database with the MCP_SERVERS configuration:
        1. Adds new servers that exist in MCP_SERVERS but not in the database
        2. Updates existing servers if their configuration has changed
        3. Removes servers from the database that are no longer in MCP_SERVERS

        Args:
            db: Database session
        """
        # Get all server configurations from the database
        db_servers = await crud_mcp_server.filter(db, limit=1000)
        db_servers_by_name = {server.name: server for server in db_servers}

        # Track servers that should exist in the database
        processed_servers = set()

        # Process servers in MCP_SERVERS configuration
        for name, config in MCP_SERVERS.items():
            processed_servers.add(name)

            try:
                # Create server if it doesn't exist in the database
                if name not in db_servers_by_name:
                    logger.info(f"Creating new MCP server configuration: {name}")
                    await crud_mcp_server.create(
                        db=db,
                        obj_in=MCPServerCreate(
                            name=name,
                            command=config["command"],
                            args=config["args"],
                            enabled=config.get("enabled", True),
                            env=config.get("env"),
                        ),
                    )
                # Update server if it exists but configuration has changed
                else:
                    existing_server = db_servers_by_name[name]
                    # Check if configuration has changed
                    if (
                        existing_server.command != config["command"]
                        or existing_server.args != config["args"]
                        or existing_server.env != config.get("env")
                    ):
                        logger.info(f"Updating MCP server configuration: {name}")
                        await crud_mcp_server.update(
                            db=db,
                            id=existing_server.id,
                            obj_in=MCPServerUpdate(
                                command=config["command"],
                                args=config["args"],
                                env=config.get("env"),
                            ),
                        )
            except Exception as e:
                logger.error(f"Failed to process MCP server {name}: {e}")

        # Remove servers that are no longer in MCP_SERVERS
        for name, server in db_servers_by_name.items():
            if name not in processed_servers:
                try:
                    logger.info(f"Removing MCP server configuration: {name}")
                    await crud_mcp_server.delete(db=db, id=server.id)
                except Exception as e:
                    logger.error(f"Failed to remove MCP server {name}: {e}")

        # Log summary
        if not db_servers:
            logger.info("Seeded database with initial MCP server configurations")
        else:
            logger.info(f"Synchronized MCP server configurations with {len(processed_servers)} servers")

        # Start enabled servers using the lifecycle manager
        try:
            await self.lifecycle_manager.start_enabled_servers()
            logger.info("Started enabled MCP servers")
        except Exception as e:
            logger.error(f"Failed to start MCP servers: {e}")


# Create a singleton instance of the service
mcp_bootstrap_service = MCPBootstrapService()
