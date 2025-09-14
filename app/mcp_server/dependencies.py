from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.dependencies import get_db_session
from app.mcp_server.service import MCPServerDomainService


async def get_mcp_server_service(db: AsyncSession = Depends(get_db_session)) -> MCPServerDomainService:
    """
    Dependency to get the MCPServerDomainService instance.
    Args:
        db (AsyncSession, optional): Database session. Defaults to Depends(get_db_session).
    Returns:
        MCPServerDomainService: Instance of the MCPServerDomainService.
    """
    return MCPServerDomainService(db=db)
