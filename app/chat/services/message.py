from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.crud import crud_message
from app.chat.exceptions.message import (
    InvalidMessageSessionException,
    InvalidParentMessageSessionException,
    MessageNotFoundException,
    ParentMessageNotFoundException,
)
from app.chat.models import ChatMessage
from app.chat.schemas import MessageCreate, MessageIn, MessageUpdate
from app.chat.services.attachment import AttachmentService
from app.chat.services.session import ChatSessionService


class ChatMessageService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.session_service = ChatSessionService(db=self.db)

    async def create_message(self, message_in: MessageIn, session_id: UUID) -> ChatMessage:
        session = await self.session_service.get_session(session_id=session_id)
        # Verify parent message if provided
        if message_in.parent_id:
            parent = await crud_message.get(db=self.db, id=message_in.parent_id)
            if not parent:
                raise ParentMessageNotFoundException(parent_id=message_in.parent_id)
            if parent.session_id != session.id:
                raise InvalidParentMessageSessionException()

        message_create = MessageCreate(**message_in.model_dump(exclude={"attachments"}))
        message = await crud_message.create_with_session(db=self.db, session_id=session_id, obj_in=message_create)

        # Process attachments if any
        if message_in.attachments:
            attachment_service = AttachmentService(db=self.db)
            await attachment_service.create_attachments(
                files=message_in.attachments, message_id=message.id, session_id=session_id
            )
            await self.db.refresh(message, ["attachments"])

        return message

    async def list_messages(self, session_id: UUID, offset: int = 0, limit: int = 10) -> Sequence[ChatMessage]:
        messages = await crud_message.list_by_session(db=self.db, session_id=session_id, offset=offset, limit=limit)
        for message in messages:
            message.usage = message.get_usage()
        return messages

    async def get_message(self, session_id: UUID, message_id: UUID) -> ChatMessage:
        message = await crud_message.get_with_attachments(self.db, id=message_id)
        session = await self.session_service.get_session(session_id=session_id)
        if not message:
            raise MessageNotFoundException(message_id=message_id)

        if message.session_id != session.id:
            raise InvalidMessageSessionException()
        message.usage = message.get_usage()
        return message

    async def update_message(self, session_id: UUID, message_id: UUID, message_in: MessageUpdate) -> ChatMessage | None:
        message = await self.get_message(session_id=session_id, message_id=message_id)
        message = await crud_message.update(db=self.db, id=message.id, obj_in=message_in)
        return message

    async def delete_message(self, session_id: UUID, message_id: UUID) -> None:
        message = await self.get_message(session_id=session_id, message_id=message_id)
        await crud_message.delete(db=self.db, id=message.id)
