class MCPServerError(Exception):
    """
    Base exception for MCP server errors.
    """

    pass


class MCPServerNotFoundError(Exception):
    """
    Raised when attempting to access a non-existent MCP server.
    """

    pass


class MCPToolError(Exception):
    """
    Raised when a tool is not found on any active MCP server.
    """

    pass
