from app.core.constants import BaseEnum


class SessionStatus(BaseEnum):
    """
    Chat session status
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
