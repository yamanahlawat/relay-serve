from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.services import ChatService
from app.database.dependencies import get_db_session


def get_chat_service(db: AsyncSession = Depends(get_db_session)) -> ChatService:
    """Get the chat service instance with database dependency."""
    return ChatService(db=db)
