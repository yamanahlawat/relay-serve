from typing import Annotated

from fastapi import Depends

from app.llm.services import ChatService


def get_chat_service() -> ChatService:
    """Get the chat service instance with database dependency."""
    return ChatService()


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
