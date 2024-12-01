from uuid import UUID

from app.core.exceptions import BaseServiceException


class ProviderNotFoundException(BaseServiceException):
    def __init__(self, provider_id: UUID):
        self.provider_id = provider_id
        super().__init__(f"Provider with id {provider_id} not found")


class DuplicateProviderException(BaseServiceException):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Provider {name} already exists")
