"""
MCP server schemas and configuration models.
"""

from .config import HTTPServerConfig, SSEServerConfig, StdioServerConfig, StreamableHTTPServerConfig
from .servers import (
    MCPServerBase,
    MCPServerCreate,
    MCPServerInDB,
    MCPServerResponse,
    MCPServerUpdate,
    ServerStatus,
)
from .tools import MCPTool

__all__ = [
    # Server schemas
    "MCPServerBase",
    "MCPServerCreate",
    "MCPServerInDB",
    "MCPServerResponse",
    "MCPServerUpdate",
    "ServerStatus",
    # Configuration schemas
    "StdioServerConfig",
    "HTTPServerConfig",
    "SSEServerConfig",
    "StreamableHTTPServerConfig",
    # Tool schemas
    "MCPTool",
]
