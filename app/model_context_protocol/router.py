from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.model_context_protocol.dependencies.server import get_mcp_server_service
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import MCPServerResponse, MCPServerToggleResponse
from app.model_context_protocol.services.server import MCPServerService

router = APIRouter(prefix="/mcp", tags=["Model Context Protocol"])


@router.get("/", response_model=list[MCPServerResponse])
async def list_mcp_servers(
    offset: int = 0,
    limit: int = 10,
    service: MCPServerService = Depends(get_mcp_server_service),
) -> list[MCPServerResponse]:
    """
    ## List MCP Servers
    List all MCP servers with their configurations and available tools.

    MCP servers are configured via code in the DEFAULT_MCP_SERVERS dictionary
    in the initialize.py file. This endpoint provides a view of the current state.

    ### Parameters
    - **offset**: Number of items to skip (default: 0)
    - **limit**: Maximum number of items to return (default: 20)

    ### Returns
    List of MCP server configurations with status and available tools
    """
    return await service.list_servers(offset=offset, limit=limit)


@router.patch("/{server_id}/toggle/", response_model=MCPServerToggleResponse)
async def toggle_mcp_server(
    server_id: UUID = Path(..., title="The ID of the MCP server"),
    service: MCPServerService = Depends(get_mcp_server_service),
) -> MCPServerToggleResponse:
    """
    ## Toggle MCP Server
    Toggle a server's enabled status. This is the only runtime modification supported.

    All other configuration changes should be made in the DEFAULT_MCP_SERVERS dictionary
    in the initialize.py file.

    ### Parameters
    - **server_id**: UUID of the MCP server

    ### Returns
    The toggled server status

    ### Raises
    - **404**: Server not found
    """
    try:
        return await service.toggle_server(server_id=server_id)
    except MCPServerError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
