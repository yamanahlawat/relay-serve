from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.model_context_protocol.constants import ServerStatus


class MCPServerBase(BaseModel):
    """
    Base schema for MCP server common attributes
    """

    command: str = Field(description="Command to execute")
    args: list[str] = Field(description="Arguments passed to command")
    enabled: bool = Field(True, description="Whether server is enabled")
    env: dict[str, str] | None = Field(default=None, description="Environment variables")


class MCPServerCreate(MCPServerBase):
    """
    Schema for creating a new MCP server
    """

    name: str = Field(description="Unique name for the server")


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

    name: str
    created_at: str
    updated_at: str

    model_config = ConfigDict(from_attributes=True)


class MCPServerResponse(MCPServerInDB):
    """
    Schema for MCP server response
    """

    status: ServerStatus = Field(ServerStatus.UNKNOWN, description="Current operational status of the server")
    available_tools: list[dict[str, Any]] = Field(default_factory=list, description="Available tools from this server")


class MCPServerToggleResponse(BaseModel):
    """
    Response schema for toggling a server
    """

    name: str
    enabled: bool
    status: ServerStatus = Field(ServerStatus.UNKNOWN, description="Current operational status of the server")
