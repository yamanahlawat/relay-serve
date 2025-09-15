from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database.crud import CRUDBase
from app.session.constants import SessionStatus
from app.session.model import ChatSession
from app.session.schema import SessionCreate, SessionUpdate


class CRUDSession(CRUDBase[ChatSession, SessionCreate, SessionUpdate]):
    """
    CRUD operations for chat sessions
    """

    async def get_with_relations(self, db: AsyncSession, id: UUID) -> ChatSession | None:
        """
        Get a chat session by ID with provider and model relations.
        Args:
            db: Database session
            id: Session ID to fetch
        Returns:
            ChatSession with relations if found, else None
        """
        query = (
            select(self.model)
            .where(self.model.id == id)
            .options(
                selectinload(self.model.provider),
                selectinload(self.model.llm_model),
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

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
