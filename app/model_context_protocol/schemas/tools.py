from typing import Any

from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel, Field


class MCPTool(BaseModel):
    """
    Represents an MCP tool with its metadata.
    """

    name: str
    description: str | None = None
    server_name: str = Field(description="Name of the MCP server providing this tool")
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """
    Represents a tool call request.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


class ToolResult(BaseModel):
    """
    Represents a tool call result.
    """

    content: list[ImageContent | TextContent | EmbeddedResource]
    call_id: str | None = None
    error: str | None = None
