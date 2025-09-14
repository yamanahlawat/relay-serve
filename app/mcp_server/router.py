from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.mcp_server.dependencies import get_mcp_server_service
from app.mcp_server.exceptions import MCPServerError
from app.mcp_server.schema import (
    MCPServerCreate,
    MCPServerResponse,
    MCPServerUpdate,
)
from app.mcp_server.service import MCPServerDomainService

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
    List of all MCP server configurations with status.
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
    The server configuration is validated before creation.

    ### Parameters
    - **server**: MCP server configuration

    ### Returns
    The created MCP server configuration with status.
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
    The updated MCP server configuration with status.

    ### Raises
    - **404**: Server not found
    - **400**: Server validation failed
    """
    try:
        return await service.update_server(server_id=server_id, model_in=server)
    except MCPServerError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


@router.delete("/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(
    server_id: UUID = Path(title="The ID of the MCP server"),
    service: MCPServerDomainService = Depends(get_mcp_server_service),
) -> None:
    """
    ## Delete MCP Server
    Delete an existing MCP server configuration.

    This endpoint allows deleting an existing MCP server configuration directly from the frontend.

    ### Parameters
    - **server_id**: UUID of the MCP server to delete

    ### Returns
    No content on success

    ### Raises
    - **404**: Server not found
    """
    try:
        await service.delete_server(server_id)
    except MCPServerError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
