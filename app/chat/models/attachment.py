from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.chat.constants import AttachmentType
from app.database.base_class import TimeStampedBase

if TYPE_CHECKING:
    from app.chat.models.message import ChatMessage


class Attachment(TimeStampedBase):
    """
    Model for storing file attachments
    """

    __tablename__ = "attachments"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    message_id: Mapped[UUID] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"))

    file_name: Mapped[str] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column()
    mime_type: Mapped[str] = mapped_column(String(100))
    type: Mapped[AttachmentType] = mapped_column(String(50))

    # URL or path to the stored file
    storage_path: Mapped[str] = mapped_column(String(500))

    # Relationship back to message
    message: Mapped["ChatMessage"] = relationship(back_populates="attachments")
