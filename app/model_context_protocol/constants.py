from app.core.constants import BaseEnum


class ServerType(BaseEnum):
    """Supported MCP server types."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class ServerStatus(BaseEnum):
    """
    Enum representing possible server operational statuses
    """

    RUNNING = "running"
    STOPPED = "stopped"
    DISABLED = "disabled"
    ERROR = "error"
    UNKNOWN = "unknown"
