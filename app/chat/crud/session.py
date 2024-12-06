from datetime import datetime, timezone
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

    async def archive(self, db: AsyncSession, id: UUID) -> ChatSession | None:
        """
        Archive a chat session.
        Args:
            db: Database session
            id: Session ID to archive
        Returns:
            Updated ChatSession if found, else None
        """
        return await self.update(
            db=db,
            id=id,
            obj_in=SessionUpdate(
                status=SessionStatus.ARCHIVED,
                extra_data={"archived_at": datetime.now(timezone.utc).isoformat()},
            ),
        )


crud_session = CRUDSession(model=ChatSession)
