import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.mcp_server.constants import ServerStatus, ServerType


class MCPServerBase(BaseModel):
    """
    Base schema for MCP server common attributes

    This class supports different MCP server types (stdio, streamable HTTP).
    The config field contains type-specific configuration.

    Examples:
    1. Stdio server:
       command: "python"
       server_type: "stdio"
       config: {
           "args": ["-m", "mcp_server_tavily"],
           "timeout": 5.0,
           "tool_prefix": "tavily_",
           "cwd": "/path/to/workdir"
       }

    2. Streamable HTTP server:
       command: "http://localhost:8000/mcp"
       server_type: "streamable_http"
       config: {
           "timeout": 15.0,
           "tool_prefix": "http_",
           "headers": {"Authorization": "Bearer token"},
           "sse_read_timeout": 300.0
       }
    """

    command: str = Field(description="Command to execute or URL for HTTP servers")
    server_type: ServerType = Field(default=ServerType.STDIO, description="Type of MCP server")
    config: dict = Field(
        default_factory=dict,
        description="Server configuration (validated by pydantic-ai when creating server instances)",
    )
    enabled: bool = Field(default=True, description="Whether server is enabled")
    env: dict[str, SecretStr] | None = Field(default=None, description="Environment variables")


class MCPServerCreate(MCPServerBase):
    """
    Schema for creating a new MCP server
    """

    name: str = Field(description="Unique name for the server")

    model_config = ConfigDict(json_encoders={SecretStr: lambda v: v.get_secret_value() if v else None})


class MCPServerUpdate(MCPServerBase):
    """
    Schema for updating an existing MCP server
    """

    command: str | None = Field(default=None, description="Command to execute or URL for HTTP servers")
    server_type: ServerType | None = Field(default=None, description="Type of MCP server")
    config: dict | None = Field(default=None, description="Server configuration")
    enabled: bool | None = Field(default=None, description="Whether server is enabled")

    model_config = ConfigDict(json_encoders={SecretStr: lambda v: v.get_secret_value() if v else None})


class MCPServerInDB(MCPServerBase):
    """
    Schema for MCP server stored in database
    """

    id: UUID
    name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class MCPServerResponse(MCPServerInDB):
    """
    Schema for MCP server response
    """

    status: ServerStatus = Field(ServerStatus.UNKNOWN, description="Current operational status of the server")
