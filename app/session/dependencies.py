from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.dependencies import get_db_session
from app.session.service import ChatSessionService


async def get_chat_session_service(db: AsyncSession = Depends(get_db_session)) -> ChatSessionService:
    """
    Get the chat session service instance with database dependency.
    """
    return ChatSessionService(db=db)
