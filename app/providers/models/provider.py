from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import TimeStampedBase
from app.providers.constants import ProviderType

if TYPE_CHECKING:
    from app.providers.models.model import LLMModel


class LLMProvider(TimeStampedBase):
    """
    Model for storing LLM provider configurations.
    """

    __tablename__ = "llm_providers"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(100), index=True)
    type: Mapped[ProviderType] = mapped_column(String(50), index=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    config: Mapped[dict] = mapped_column(type_=JSONB, default=dict)

    # Relationships
    models: Mapped[list["LLMModel"]] = relationship(back_populates="provider", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("name", name="uq_provider_name"),)
