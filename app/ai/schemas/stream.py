from datetime import datetime, timezone
from typing import Any

from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel, Field

from app.chat.schemas.message import MessageRead
from app.core.constants import BaseEnum


class StreamBlockType(BaseEnum):
    """
    Types of blocks in the message stream
    """

    CONTENT = "content"
    ERROR = "error"

    TOOL_START = "tool_start"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    THINKING = "thinking"
    DONE = "done"


class StreamBlock(BaseModel):
    """
    Base model for different types of stream blocks.
    - For thinking/content blocks: use 'content' field
    - For tool blocks: use tool_* fields
    - For tool argument streaming: use 'args_delta' field for chunks, 'tool_args' for final args
    - For errors: use error_* fields
    - For completion: use usage field
    """

    type: StreamBlockType

    # Used for thinking and regular content blocks
    content: str | None = None

    # Tool-specific fields
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None
    tool_result: list[TextContent | ImageContent | EmbeddedResource] | None = None

    # For streaming tool arguments - contains delta chunks as they stream
    args_delta: str | None = None

    # Error handling
    error_type: str | None = None
    error_detail: str | None = None

    # Final message
    message: MessageRead | None = None

    # Usage information (for completion blocks)
    usage: dict[str, Any] | None = None

    # Timestamp for tracking when the block was created
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
