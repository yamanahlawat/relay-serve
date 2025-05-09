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
from app.model_context_protocol.schemas.servers import MCPServerCreate


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

        Args:
            db: Database session
        """
        # Check if there are any server configurations in the database using CRUD
        db_servers = await crud_mcp_server.filter(db, limit=10)

        # If no servers in DB, seed with defaults
        if not db_servers:
            logger.info("No MCP servers found in database. Seeding defaults...")
            for name, config in MCP_SERVERS.items():
                try:
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
                    logger.info(f"Created MCP server configuration: {name}")
                except Exception as e:
                    logger.error(f"Failed to create MCP server {name}: {e}")

        # Start enabled servers using the lifecycle manager
        try:
            await self.lifecycle_manager.start_enabled_servers()
            logger.info("Started enabled MCP servers")
        except Exception as e:
            logger.error(f"Failed to start MCP servers: {e}")


# Create a singleton instance of the service
mcp_bootstrap_service = MCPBootstrapService()
