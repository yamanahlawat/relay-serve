"""Database model for LLM models."""

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import TimeStampedBase
from app.llms.models.provider import LLMProvider

if TYPE_CHECKING:
    from app.chat.models.session import ChatSession


class LLMModel(TimeStampedBase):
    """
    Database model for an LLM model.

    Stores user-configured models that are available from their configured providers.
    Users manually add models through the UI after configuring providers.
    Model names should match what pydantic_ai expects (e.g., 'gpt-4o', 'claude-3-5-sonnet-latest').
    """

    __tablename__ = "llm_model"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    # Model name as used by pydantic_ai (e.g., 'gpt-4o', 'claude-3-5-sonnet-latest')
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Whether this model is active/enabled
    is_active: Mapped[bool] = mapped_column(default=True)

    # Context length/window (optional, for user reference)
    context_length: Mapped[int | None] = mapped_column(nullable=True)

    # Default model settings (can be overridden per chat session)
    default_temperature: Mapped[float] = mapped_column(default=0.7)
    default_max_tokens: Mapped[int | None] = mapped_column(default=None)
    default_top_p: Mapped[float] = mapped_column(default=0.9)

    # Foreign Keys
    provider_id: Mapped[UUID] = mapped_column(ForeignKey(LLMProvider.id))

    # Relationships
    provider: Mapped[LLMProvider] = relationship(back_populates="models")
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="llm_model", cascade="all, delete-orphan")

    @property
    def pydantic_ai_model_string(self) -> str:
        """
        Generate the pydantic_ai model string format.
        Returns:
            String in format "provider_type:model" (e.g., "openai:gpt-4o")
        """
        return f"{self.provider.provider_type.value}:{self.name}"

    __table_args__ = (UniqueConstraint("provider_id", "name", name="uq_provider_model"),)
