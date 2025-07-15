from app.ai.services import ChatService


def get_chat_service() -> ChatService:
    """Get the chat service instance with database dependency."""
    return ChatService()
