from sqlalchemy.ext.asyncio import AsyncSession


class ChatService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
