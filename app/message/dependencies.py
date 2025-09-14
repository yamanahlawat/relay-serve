from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.dependencies import get_db_session
from app.message.service import ChatMessageService


async def get_chat_message_service(db: AsyncSession = Depends(get_db_session)) -> ChatMessageService:
    """
    Get the chat message service instance with database dependency.
    """
    return ChatMessageService(db=db)
