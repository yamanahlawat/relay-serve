from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, remote

from app.core.database.base_class import TimeStampedBase
from app.message.constants import MessageRole, MessageStatus

from app.attachment.model import Attachment

if TYPE_CHECKING:
    from app.session.model import ChatSession


class ChatMessage(TimeStampedBase):
    """
    Model for chat messages
    """

    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[MessageRole] = mapped_column(String(50))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[MessageStatus] = mapped_column(String(50), default=MessageStatus.PENDING, index=True)

    # Token usage tracking
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    input_cost: Mapped[float] = mapped_column(Numeric(12, 8), default=0.0)
    output_cost: Mapped[float] = mapped_column(Numeric(12, 8), default=0.0)

    # Message threading
    parent_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True)

    # Attachments
    attachments: Mapped[list["MessageAttachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", lazy="selectin"
    )

    # Direct relationship to attachments through association
    direct_attachments: Mapped[list[Attachment]] = relationship(
        secondary="message_attachments", viewonly=True, lazy="selectin"
    )

    # Error tracking
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Additional data (e.g., function calls, citations)
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    parent: Mapped["ChatMessage | None"] = relationship(
        "ChatMessage",
        remote_side=[remote(id)],
        backref="children",
    )

    __table_args__ = (Index("ix_chat_messages_session_created", "session_id", "created_at", postgresql_using="btree"),)

    def get_usage(self) -> dict[str, Any]:
        """
        Get usage metrics for the message
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "total_cost": float(self.input_cost + self.output_cost),
        }


class MessageAttachment(TimeStampedBase):
    """
    Model for storing attachments linked to chat messages
    """

    __tablename__ = "message_attachments"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attachment_id: Mapped[UUID] = mapped_column(
        ForeignKey(Attachment.id, ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationship to the parent message
    message: Mapped[ChatMessage] = relationship(back_populates="attachments")
    # Reference to the attachment (one-directional – Attachment remains unaware)
    attachment: Mapped[Attachment] = relationship()
