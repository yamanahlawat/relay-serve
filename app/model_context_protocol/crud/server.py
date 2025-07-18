from app.database.crud import CRUDBase
from app.model_context_protocol.models.server import MCPServer
from app.model_context_protocol.schemas.servers import MCPServerCreate, MCPServerUpdate


class CRUDMCPServer(CRUDBase[MCPServer, MCPServerCreate, MCPServerUpdate]):
    """
    CRUD operations for MCP server configurations.
    """

    pass


# Create a singleton instance of the CRUD class
crud_mcp_server = CRUDMCPServer(model=MCPServer)
