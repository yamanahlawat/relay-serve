from uuid import UUID

from app.core.exceptions import BaseServiceException


class MessageNotFoundException(BaseServiceException):
    def __init__(self, message_id: UUID) -> None:
        self.message_id = message_id
        self.message = f"Message with id {message_id} not found"
        super().__init__(self.message)


class ParentMessageNotFoundException(BaseServiceException):
    def __init__(self, parent_id: UUID) -> None:
        self.parent_id = parent_id
        self.message = f"Parent message {parent_id} not found"
        super().__init__(self.message)


class InvalidParentMessageSessionException(BaseServiceException):
    def __init__(self) -> None:
        self.message = "Parent message belongs to a different session"
        super().__init__(self.message)


class InvalidMessageSessionException(BaseServiceException):
    def __init__(self) -> None:
        self.message = "Message belongs to a different session"
        super().__init__(self.message)
