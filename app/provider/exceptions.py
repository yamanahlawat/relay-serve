from uuid import UUID

from app.core.exceptions import BaseServiceException


class ProviderNotFoundException(BaseServiceException):
    def __init__(self, provider_id: UUID) -> None:
        self.provider_id = provider_id
        self.message = f"Provider with id {provider_id} not found"
        super().__init__(self.message)


class DuplicateProviderException(BaseServiceException):
    def __init__(self, name: str) -> None:
        self.name = name
        self.message = f"Provider {name} already exists"
        super().__init__(self.message)
