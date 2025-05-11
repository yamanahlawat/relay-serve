from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.model_context_protocol.crud.server import crud_mcp_server
from app.model_context_protocol.dependencies.server import get_mcp_server_service
from app.model_context_protocol.exceptions import MCPServerError
from app.model_context_protocol.schemas.servers import (
    MCPServerCreate,
    MCPServerResponse,
    MCPServerUpdate,
)
from app.model_context_protocol.services.domain import MCPServerDomainService

router = APIRouter(prefix="/mcp", tags=["Model Context Protocol"])


@router.get("/", response_model=list[MCPServerResponse])
async def list_mcp_servers(
    offset: int = 0,
    limit: int = 10,
    service: MCPServerDomainService = Depends(get_mcp_server_service),
) -> list[MCPServerResponse]:
    """
    ## List All MCP Servers
    List all configured MCP servers with their configurations and statuses.

    ### Parameters
    - **offset**: Number of items to skip (default: 0)
    - **limit**: Maximum number of items to return (default: 10)

    ### Returns
    List of all MCP server configurations with status and available tools
    """
    return await service.list_servers(offset=offset, limit=limit)


@router.post("/", response_model=MCPServerResponse)
async def create_mcp_server(
    server: MCPServerCreate,
    service: MCPServerDomainService = Depends(get_mcp_server_service),
) -> MCPServerResponse:
    """
    ## Create MCP Server
    Create a new MCP server configuration.

    This endpoint allows creating new MCP server configurations directly from the frontend.

    ### Parameters
    - **server**: MCP server configuration

    ### Returns
    The created MCP server configuration with status
    """
    return await service.create_server(server)


@router.put("/{server_id}", response_model=MCPServerResponse)
async def update_mcp_server(
    server: MCPServerUpdate,
    server_id: UUID = Path(title="The ID of the MCP server"),
    service: MCPServerDomainService = Depends(get_mcp_server_service),
) -> MCPServerResponse:
    """
    ## Update MCP Server
    Update an existing MCP server configuration.

    This endpoint allows updating an existing MCP server configuration directly from the frontend.

    ### Parameters
    - **server_id**: UUID of the MCP server
    - **server**: MCP server configuration update

    ### Returns
    The updated MCP server configuration with status

    ### Raises
    - **404**: Server not found
    """
    try:
        return await service.update_server(server_id, server)
    except MCPServerError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


@router.patch("/{server_id}/toggle/", response_model=MCPServerResponse)
async def toggle_mcp_server(
    server_id: UUID = Path(title="The ID of the MCP server"),
    service: MCPServerDomainService = Depends(get_mcp_server_service),
) -> MCPServerResponse:
    """
    ## Toggle MCP Server
    Toggle a server's enabled status.

    This endpoint toggles the enabled status of a server by using the update_server method with toggle=True.

    ### Parameters
    - **server_id**: UUID of the MCP server

    ### Returns
    The toggled server status

    ### Raises
    - **404**: Server not found
    """
    try:
        # Get the server from database to determine current enabled status
        existing = await crud_mcp_server.get(db=service.db, id=server_id)
        if not existing:
            raise MCPServerError("Server not found")

        # Create update with toggled enabled status
        update_data = MCPServerUpdate(enabled=not existing.enabled)

        # Use update_server with toggle=True to get MCPServerToggleResponse
        return await service.update_server(server_id=server_id, update_data=update_data, toggle=True)
    except MCPServerError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
