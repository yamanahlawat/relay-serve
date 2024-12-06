from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.services.message import ChatMessageService
from app.database.dependencies import get_db_session


async def get_chat_message_service(db: AsyncSession = Depends(get_db_session)) -> ChatMessageService:
    return ChatMessageService(db=db)
