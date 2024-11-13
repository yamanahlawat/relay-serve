from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
        db_obj = await self.get(db=db, id=id)
        if db_obj and db_obj.status == SessionStatus.ACTIVE:
            return db_obj
        return None

    async def list_by_status(
        self,
        db: AsyncSession,
        status: SessionStatus,
        offset: int = 0,
        limit: int = 10,
    ) -> Sequence[ChatSession]:
        """
        List chat sessions by status.
        Args:
            db: Database session
            status: Session status to filter by
            offset: Number of records to skip
            limit: Maximum number of records to return
        Returns:
            List of chat sessions
        """
        query = (
            select(self.model)
            .where(self.model.status == status)
            .order_by(self.model.last_message_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

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
