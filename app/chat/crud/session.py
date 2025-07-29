from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.chat.constants import SessionStatus
from app.chat.models import ChatSession
from app.chat.schemas.session import SessionCreate, SessionUpdate
from app.database.crud import CRUDBase


class CRUDSession(CRUDBase[ChatSession, SessionCreate, SessionUpdate]):
    """
    CRUD operations for chat sessions
    """

    async def get_active(self, db: AsyncSession, id: UUID) -> ChatSession | None:
        """
        Get an active chat session by ID.
        Args:
            db: Database session
            id: Session ID to fetch
        Returns:
            Active ChatSession if found, else None
        """
        query = (
            select(self.model)
            .where(self.model.id == id, self.model.status == SessionStatus.ACTIVE)
            .options(
                selectinload(self.model.provider),
                selectinload(self.model.llm_model),
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()


crud_session = CRUDSession(model=ChatSession)
