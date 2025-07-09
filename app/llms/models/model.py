from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.chat.constants import llm_defaults
from app.database.base_class import TimeStampedBase
from app.llms.models.provider import LLMProvider

if TYPE_CHECKING:
    from app.chat.models.session import ChatSession


class LLMModel(TimeStampedBase):
    """
    Model for storing LLM model configurations.
    """

    __tablename__ = "llm_models"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(default=True)

    # Model configuration
    default_max_tokens: Mapped[int | None] = mapped_column(default=llm_defaults.MAX_TOKENS)
    default_temperature: Mapped[float] = mapped_column(default=llm_defaults.TEMPERATURE)
    default_top_p: Mapped[float] = mapped_column(default=llm_defaults.TOP_P)

    # Foreign Keys
    provider_id: Mapped[UUID] = mapped_column(ForeignKey(LLMProvider.id))

    # Relationships
    provider: Mapped[LLMProvider] = relationship(back_populates="models")
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="llm_model", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("provider_id", "name", name="uq_provider_model"),)
