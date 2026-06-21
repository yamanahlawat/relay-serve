from typing import Annotated

from fastapi import Depends

from app.core.database.dependencies import DBSessionDep
from app.session.service import ChatSessionService


async def get_chat_session_service(db: DBSessionDep) -> ChatSessionService:
    """
    Get the chat session service instance with database dependency.
    """
    return ChatSessionService(db=db)


ChatSessionServiceDep = Annotated[ChatSessionService, Depends(get_chat_session_service)]
