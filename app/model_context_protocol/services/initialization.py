"""
MCP Server Initialization Service

This service is responsible for initializing MCP servers from the database
or seeding default configurations if the database is empty.
"""

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.initialize import MCP_SERVERS, mcp_registry
from app.model_context_protocol.schemas.servers import MCPServerCreate


class MCPInitializationService:
    """
    Service for initializing MCP servers.

    Follows domain-driven design principles by encapsulating the initialization
    logic in a dedicated service class.
    """

    async def initialize_servers(self) -> None:
        """
        Initialize MCP servers from database or seed defaults.

        This method:
        1. Checks if there are any server configurations in the database
        2. Seeds default configurations if the database is empty
        3. Starts enabled servers
        """
        async with AsyncSessionLocal() as db:
            await self._do_initialize(db)

    async def _do_initialize(self, db: AsyncSession) -> None:
        """
        Internal method to perform the actual initialization with a database session.

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

        # Start enabled servers using the registry
        try:
            await mcp_registry.start_enabled_servers()
            logger.info("Started enabled MCP servers")
        except Exception as e:
            logger.error(f"Failed to start MCP servers: {e}")


# Create a singleton instance of the service
mcp_init_service = MCPInitializationService()
