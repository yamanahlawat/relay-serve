from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import TimeStampedBase
from app.llms.constants import ProviderType

if TYPE_CHECKING:
    from app.chat.models.session import ChatSession
    from app.llms.models.model import LLMModel


class LLMProvider(TimeStampedBase):
    """
    Model for storing LLM provider configurations.
    """

    __tablename__ = "llm_providers"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String(200), index=True)
    type: Mapped[ProviderType] = mapped_column(index=True)
    is_active: Mapped[bool] = mapped_column(default=True)

    # User's API key for this provider (nullable for providers that don't need keys)
    api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Custom base URL (for OpenAI-compatible APIs, local models, etc.)
    base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationships
    models: Mapped[list["LLMModel"]] = relationship(back_populates="provider", cascade="all, delete-orphan")
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="provider", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("name", name="uq_provider_name"),)
