from typing import Annotated

from fastapi import Depends

from app.core.database.dependencies import DBSessionDep
from app.message.service import ChatMessageService


async def get_chat_message_service(db: DBSessionDep) -> ChatMessageService:
    """
    Get the chat message service instance with database dependency.
    """
    return ChatMessageService(db=db)


ChatMessageServiceDep = Annotated[ChatMessageService, Depends(get_chat_message_service)]
