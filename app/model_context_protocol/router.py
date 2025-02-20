from fastapi import APIRouter, Depends

from app.model_context_protocol.schemas.servers import MCPServerTools
from app.model_context_protocol.schemas.tools import BaseTool
from app.model_context_protocol.services.tool import MCPToolService

router = APIRouter(prefix="/mcp", tags=["Model Context Protocol"])


@router.get("/", response_model=list[MCPServerTools])
async def list_mcp_tools(
    tool_service: MCPToolService = Depends(MCPToolService),
) -> list[MCPServerTools]:
    """
    List all available MCP tools grouped by server.
    Returns:
        List of servers and their available tools.
    """
    # Get all available tools using existing service
    tools = await tool_service.get_available_tools(refresh=True)

    # Group tools by server
    servers_dict: dict[str, list[BaseTool]] = {}
    for tool in tools:
        servers_dict.setdefault(tool.server_name, []).append(tool)
    # Format response
    return [MCPServerTools(name=server_name, tools=server_tools) for server_name, server_tools in servers_dict.items()]
