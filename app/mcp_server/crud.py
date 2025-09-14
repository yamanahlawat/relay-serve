from app.core.database.crud import CRUDBase
from app.mcp_server.model import MCPServer
from app.mcp_server.schema import MCPServerCreate, MCPServerUpdate


class CRUDMCPServer(CRUDBase[MCPServer, MCPServerCreate, MCPServerUpdate]):
    """
    CRUD operations for MCP server configurations.
    """

    pass


# Create a singleton instance of the CRUD class
crud_mcp_server = CRUDMCPServer(model=MCPServer)
