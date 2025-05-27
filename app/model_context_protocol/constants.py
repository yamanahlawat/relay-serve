from enum import Enum


class ServerStatus(str, Enum):
    """
    Enum representing possible server operational statuses
    """

    RUNNING = "running"
    STOPPED = "stopped"
    DISABLED = "disabled"
    ERROR = "error"
    UNKNOWN = "unknown"


class MCPEventType(str, Enum):
    """
    Enum representing event types for the MCP event bus system.
    These events are used for communication between decoupled components.
    """

    SERVER_STARTED = "mcp_server_started"  # When a server is started successfully
    SERVER_SHUTDOWN = "mcp_server_shutdown"  # When a server is shut down
    SERVER_ERROR = "mcp_server_error"  # When a server encounters an error
    TOOL_REGISTERED = "mcp_tool_registered"  # When new tools are registered
    TOOL_EXECUTED = "mcp_tool_executed"  # When a tool is executed
