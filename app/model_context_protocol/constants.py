from enum import Enum


class ServerStatus(str, Enum):
    """
    Enum representing possible server operational statuses
    """

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"
