from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.crud import CRUDBase
from app.model_context_protocol.models.server import MCPServer
from app.model_context_protocol.schemas.servers import MCPServerCreate, MCPServerUpdate


class CRUDMCPServer(CRUDBase[MCPServer, MCPServerCreate, MCPServerUpdate]):
    """
    CRUD operations for MCP server configurations.
    """

    async def get_by_name(self, db: AsyncSession, *, name: str) -> MCPServer | None:
        """
        Get a server by its name.
        Args:
            db: Database session
            name: Name of the server to retrieve
        Returns:
            The server if found, None otherwise
        """
        query = select(self.model).where(self.model.name == name)
        result = await db.execute(query)
        return result.scalar_one_or_none()


# Create a singleton instance of the CRUD class
crud_mcp_server = CRUDMCPServer(model=MCPServer)
