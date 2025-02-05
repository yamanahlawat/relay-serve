from typing import Sequence
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.chat.constants import MessageRole, MessageStatus
from app.chat.models import ChatMessage
from app.chat.schemas.message import MessageCreate, MessageUpdate
from app.database.crud import CRUDBase


class CRUDMessage(CRUDBase[ChatMessage, MessageCreate, MessageUpdate]):
    """
    CRUD operations for chat messages
    """

    async def get_with_attachments(self, db: AsyncSession, id: UUID) -> ChatMessage | None:
        """
        Get a chat message by ID with attachments.
        Args:
            db (AsyncSession): Database session
            id (UUID): ID of the message to fetch
        Returns:
            ChatMessage | None: Chat message with attachments if found, else None
        """
        stmt = select(self.model).options(selectinload(self.model.attachments)).where(self.model.id == id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_with_session(
        self,
        db: AsyncSession,
        *,
        session_id: UUID,
        obj_in: MessageCreate,
    ) -> ChatMessage:
        """
        Create a new message for a specific chat session.
        Args:
            db: Database session
            session_id: ID of the chat session
            obj_in: Message data to create
        Returns:
            Created ChatMessage
        """
        message_data = obj_in.model_dump()
        usage = message_data.pop("usage")
        db_obj = ChatMessage(
            **message_data,
            session_id=session_id,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            input_cost=usage["input_cost"],
            output_cost=usage["output_cost"],
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def list_by_session(
        self,
        db: AsyncSession,
        session_id: UUID,
        offset: int = 0,
        limit: int = 10,
    ) -> Sequence[ChatMessage]:
        """
        List messages for a specific chat session.
        Args:
            db: Database session
            session_id: ID of the chat session
            offset: Number of records to skip
            limit: Maximum number of records to return
        Returns:
            List of chat messages
        """
        query = (
            select(self.model)
            .options(selectinload(self.model.attachments))
            .where(self.model.session_id == session_id)
            .order_by(desc(self.model.created_at))
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def get_session_context(
        self,
        db: AsyncSession,
        session_id: UUID,
        exclude_message_id: UUID | None = None,
    ) -> Sequence[ChatMessage]:
        """
        Get recent context messages for a chat session.
        Args:
            db: Database session
            session_id: ID of the chat session
            limit: Maximum number of recent messages to return
        Returns:
            List of recent messages for context
        """
        conditions = [
            self.model.session_id == session_id,
            self.model.role.in_([MessageRole.USER, MessageRole.ASSISTANT]),
            self.model.status == MessageStatus.COMPLETED,
        ]

        if exclude_message_id:
            conditions.append(self.model.id != exclude_message_id)

        query = select(self.model).where(*conditions).order_by(self.model.created_at.asc())

        result = await db.execute(query)
        return result.scalars().all()


crud_message = CRUDMessage(model=ChatMessage)
