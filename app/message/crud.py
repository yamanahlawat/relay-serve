from typing import Sequence
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database.crud import CRUDBase
from app.message.constants import MessageRole, MessageStatus
from app.message.model import ChatMessage, MessageAttachment
from app.message.schema import MessageCreate, MessageUpdate


class CRUDMessage(CRUDBase[ChatMessage, MessageCreate, MessageUpdate]):
    """
    CRUD operations for chat messages
    """

    async def get_with_attachments(self, db: AsyncSession, id: UUID) -> ChatMessage:
        """
        Get a chat message by ID with attachments.
        Args:
            db (AsyncSession): Database session
            id (UUID): ID of the message to fetch
        Returns:
            ChatMessage: Chat message with attachments
        """
        statement = select(self.model).options(selectinload(self.model.direct_attachments)).where(self.model.id == id)
        result = await db.execute(statement)
        return result.scalar_one()

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: MessageCreate,
        session_id: UUID,
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
        attachments = message_data.pop("attachment_ids")
        db_obj = ChatMessage(
            **message_data,
            session_id=session_id,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            input_cost=usage["input_cost"],
            output_cost=usage["output_cost"],
        )
        db.add(db_obj)
        await db.flush()

        # Create message attachments
        if attachments:
            message_attachments = [
                MessageAttachment(message_id=db_obj.id, attachment_id=attach_id) for attach_id in attachments
            ]
            db.add_all(message_attachments)

        await db.commit()
        # Refresh with explicit loading of relationships
        db_obj = await self.get_with_attachments(db=db, id=db_obj.id)
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
            .options(selectinload(self.model.direct_attachments))
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

        query = (
            select(self.model)
            .options(selectinload(self.model.attachments))
            .where(*conditions)
            .order_by(self.model.created_at.asc())
        )

        result = await db.execute(query)
        return result.scalars().all()


crud_message = CRUDMessage(model=ChatMessage)
