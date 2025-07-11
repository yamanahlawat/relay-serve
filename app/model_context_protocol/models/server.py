from uuid import UUID, uuid4

from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base_class import TimeStampedBase
from app.model_context_protocol.constants import ServerType


class MCPServer(TimeStampedBase):
    """
    Database model for MCP server configurations.

    This model stores server configurations that can be managed by users
    through the frontend interface. It allows enabling, disabling, and
    configuring different MCP servers dynamically.

    Supports multiple server types:
    - stdio: Traditional process-based servers
    - sse: HTTP with Server-Sent Events
    - streamable_http: HTTP with Streamable transport
    """

    __tablename__ = "mcp_servers"

    # Primary identifier - UUID
    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)

    # Server name - unique identifier
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    # Server type - determines how the server is connected
    server_type: Mapped[ServerType] = mapped_column(default=ServerType.STDIO, nullable=False, index=True)

    # Server command (for stdio servers) or URL (for HTTP servers)
    command: Mapped[str] = mapped_column(String(500), nullable=False)

    # Server configuration (flexible JSON structure)
    # For stdio: {"args": [...], "cwd": "...", "timeout": 5.0, "tool_prefix": "..."}
    # For sse/http: {"timeout": 5.0, "tool_prefix": "...", "headers": {...}}
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Server status
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Environment variables (for stdio servers, securely stored)
    env: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
