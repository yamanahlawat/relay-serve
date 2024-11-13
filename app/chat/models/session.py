from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, column_property, mapped_column, relationship
from sqlalchemy.sql import func, select

from app.chat.constants import SessionStatus
from app.chat.models.message import ChatMessage
from app.database.base_class import TimeStampedBase
from app.providers.models import LLMModel, LLMProvider


class ChatSession(TimeStampedBase):
    """
    Model for chat sessions
    """

    __tablename__ = "chat_sessions"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    title: Mapped[str] = mapped_column(String(255))
    status: Mapped[SessionStatus] = mapped_column(String(50), default=SessionStatus.ACTIVE, index=True)

    # System context/instructions for the chat
    system_context: Mapped[str | None] = mapped_column(nullable=True)

    # Provider configuration
    provider_id: Mapped[UUID] = mapped_column(ForeignKey(LLMProvider.id, ondelete="RESTRICT"))
    llm_model_id: Mapped[UUID] = mapped_column(ForeignKey(LLMModel.id, ondelete="RESTRICT"))

    # Additional data
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    last_message_at: Mapped[datetime | None] = mapped_column(nullable=True, index=True)

    # Relationships
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by=ChatMessage.created_at
    )

    # Computed properties for totals
    total_input_tokens = column_property(
        select(func.coalesce(func.sum(ChatMessage.input_tokens), 0))
        .where(ChatMessage.session_id == id)
        .scalar_subquery()
    )

    total_output_tokens: Mapped[int] = column_property(
        select(func.coalesce(func.sum(ChatMessage.output_tokens), 0))
        .where(ChatMessage.session_id == id)
        .scalar_subquery()
    )

    total_input_cost: Mapped[float] = column_property(
        select(func.coalesce(func.sum(ChatMessage.input_cost), 0.0))
        .where(ChatMessage.session_id == id)
        .scalar_subquery()
    )

    total_output_cost: Mapped[float] = column_property(
        select(func.coalesce(func.sum(ChatMessage.output_cost), 0.0))
        .where(ChatMessage.session_id == id)
        .scalar_subquery()
    )

    __table_args__ = (Index("ix_chat_sessions_status_last_message", status, last_message_at.desc()),)

    @hybrid_property
    def total_cost(self) -> float:
        """
        Total cost of the session
        """
        return self.total_input_cost + self.total_output_cost

    @total_cost.expression
    def total_cost(cls):
        """
        SQL expression for total cost
        """
        return cls.total_input_cost + cls.total_output_cost

    # Add helper method for usage
    def get_usage(self) -> dict[str, Any]:
        """
        Get usage metrics for the session
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost": float(self.total_input_cost),
            "output_cost": float(self.total_output_cost),
            "total_cost": float(self.total_cost),
        }
