import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.model_context_protocol.constants import ServerStatus
from app.model_context_protocol.schemas.tools import MCPTool


class MCPServerBase(BaseModel):
    """
    Base schema for MCP server common attributes

    This class supports both direct command execution and Docker-based MCP servers.
    For Docker-based servers, the command should be 'docker' and args should include
    the 'run' command and necessary parameters.

    Examples:
    1. Direct execution:
       command: "python"
       args: ["-m", "mcp_server_tavily"]

    2. Docker Gateway:
       command: "docker"
       args: ["run", "--rm", "-i", "alpine/socat", "STDIO", "TCP:host.docker.internal:8811"]

    3. Docker Container:
       command: "docker"
       args: ["run", "-i", "--rm", "-e", "API_KEY", "mcp/server-name"]
       env: {"API_KEY": "your-api-key"}
    """

    command: str = Field(description="Command to execute")
    args: list[str] = Field(description="Arguments passed to command")
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

    command: str | None = Field(default=None, description="Command to execute")
    args: list[str] | None = Field(default=None, description="Arguments passed to command")
    enabled: bool | None = Field(default=None, description="Whether server is enabled")


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
    available_tools: list[MCPTool] = Field(default_factory=list, description="Available tools from this server")
