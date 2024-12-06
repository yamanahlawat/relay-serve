from uuid import UUID

from app.core.exceptions import BaseServiceException


class SessionNotFoundException(BaseServiceException):
    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        self.message = f"Session with id {session_id} not found"
        super().__init__(self.message)
