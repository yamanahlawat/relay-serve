from uuid import UUID

from app.core.exceptions import BaseServiceException


class ModelNotFoundException(BaseServiceException):
    def __init__(self, model_id: UUID) -> None:
        self.model_id = model_id
        self.message = f"Model with id {model_id} not found"
        super().__init__(self.message)


class DuplicateModelException(BaseServiceException):
    def __init__(self, name: str) -> None:
        self.name = name
        self.message = f"Model {name} already exists for this provider"
        super().__init__(self.message)


class InvalidModelProviderException(BaseServiceException):
    def __init__(self) -> None:
        self.message = "Model does not belong to the specified provider"
        super().__init__(self.message)
