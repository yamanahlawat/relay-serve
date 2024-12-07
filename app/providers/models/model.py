from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.chat.constants import llm_defaults
from app.database.base_class import TimeStampedBase
from app.providers.models.provider import LLMProvider

if TYPE_CHECKING:
    from app.chat.models.session import ChatSession


class LLMModel(TimeStampedBase):
    """
    Model for storing LLM model configurations.
    """

    __tablename__ = "llm_models"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    provider_id: Mapped[UUID] = mapped_column(ForeignKey(LLMProvider.id))
    name: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(default=True)

    # Model configuration
    max_tokens: Mapped[int | None] = mapped_column(default=None)
    temperature: Mapped[float] = mapped_column(default=llm_defaults.TEMPERATURE)
    top_p: Mapped[float] = mapped_column(default=llm_defaults.TOP_P)
    config: Mapped[dict] = mapped_column(type_=JSONB, default=dict)

    # Cost tracking (in USD)
    input_cost_per_token: Mapped[float] = mapped_column(default=0.0)
    output_cost_per_token: Mapped[float] = mapped_column(default=0.0)

    # Relationships
    provider: Mapped[LLMProvider] = relationship(back_populates="models")
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="llm_model", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("provider_id", "name", name="uq_provider_model"),)
