from typing import Annotated

from fastapi import Depends

from app.core.database.dependencies import DBSessionDep
from app.mcp_server.service import MCPServerDomainService


async def get_mcp_server_service(db: DBSessionDep) -> MCPServerDomainService:
    """
    Get the MCPServerDomainService instance with database dependency.
    """
    return MCPServerDomainService(db=db)


MCPServerServiceDep = Annotated[MCPServerDomainService, Depends(get_mcp_server_service)]
