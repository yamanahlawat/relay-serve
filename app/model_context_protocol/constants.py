from enum import Enum


class ServerStatus(str, Enum):
    """
    Enum representing possible server operational statuses
    """

    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"
