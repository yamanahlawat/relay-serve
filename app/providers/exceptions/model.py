from uuid import UUID

from app.core.exceptions import BaseServiceException


class ModelNotFoundException(BaseServiceException):
    def __init__(self, model_id: UUID) -> None:
        self.model_id = model_id
        super().__init__(f"Model with id {model_id} not found")


class DuplicateModelException(BaseServiceException):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Model {name} already exists for this provider")
