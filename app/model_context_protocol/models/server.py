from uuid import UUID, uuid4

from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base_class import TimeStampedBase


class MCPServer(TimeStampedBase):
    """
    Database model for MCP server configurations.

    This model stores server configurations that can be managed by users
    through the frontend interface. It allows enabling, disabling, and
    configuring different MCP servers dynamically.
    """

    __tablename__ = "mcp_servers"

    # Primary identifier - UUID
    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    # Server name - unique identifier
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    # Server command and arguments
    command: Mapped[str] = mapped_column(String(255), nullable=False)
    args: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Server status
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Environment variables (securely stored)
    env: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
