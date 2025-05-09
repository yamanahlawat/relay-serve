from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dependencies import get_db_session
from app.model_context_protocol.services.server import MCPServerService


async def get_mcp_server_service(db: AsyncSession = Depends(get_db_session)) -> MCPServerService:
    """
    Dependency to get the MCPServerService instance.
    Args:
        db (AsyncSession, optional): Database session. Defaults to Depends(get_db_session).
    Returns:
        MCPServerService: Instance of the MCPServerService.
    """
    return MCPServerService(db=db)
