"""
Configuration validation schemas for different MCP server types.
"""

from pydantic import BaseModel, Field


class StdioServerConfig(BaseModel):
    """Configuration schema for STDIO MCP servers."""

    args: list[str] = Field(default_factory=list, description="Command arguments")
    timeout: float = Field(default=5.0, gt=0, description="Connection timeout in seconds")
    tool_prefix: str | None = Field(default=None, description="Prefix for tool names")
    cwd: str | None = Field(default=None, description="Working directory for the command")


class HTTPServerConfig(BaseModel):
    """Configuration schema for HTTP-based MCP servers (SSE and Streamable HTTP)."""

    timeout: float = Field(default=5.0, gt=0, description="Connection timeout in seconds")
    tool_prefix: str | None = Field(default=None, description="Prefix for tool names")


# Aliases for clarity
SSEServerConfig = HTTPServerConfig
StreamableHTTPServerConfig = HTTPServerConfig
