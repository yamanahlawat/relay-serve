from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.database.base_class import TimeStampedBase
from app.providers.models.provider import LLMProvider


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
    temperature: Mapped[float] = mapped_column(default=settings.DEFAULT_TEMPERATURE)
    top_p: Mapped[float] = mapped_column(default=settings.DEFAULT_TOP_P)
    config: Mapped[dict] = mapped_column(type_=JSONB, default=dict)

    # Cost tracking (in USD)
    input_cost_per_token: Mapped[float] = mapped_column(default=0.0)
    output_cost_per_token: Mapped[float] = mapped_column(default=0.0)

    # Relationships
    provider: Mapped[LLMProvider] = relationship(back_populates="models")

    __table_args__ = (UniqueConstraint("provider_id", "name", name="uq_provider_model"),)
