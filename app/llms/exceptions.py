"""Exceptions for LLM provider and model operations."""


class LLMProviderException(Exception):
    """Base exception for LLM provider operations."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ProviderNotFoundException(LLMProviderException):
    """Exception raised when a provider is not found."""

    def __init__(self, provider_id: str) -> None:
        super().__init__(f"Provider with id {provider_id} not found")


class DuplicateProviderException(LLMProviderException):
    """Exception raised when trying to create a provider with a duplicate name."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(f"Provider with name {provider_name} already exists")


class LLMModelException(Exception):
    """Base exception for LLM model operations."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ModelNotFoundException(LLMModelException):
    """Exception raised when a model is not found."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model with id {model_id} not found")


class DuplicateModelException(LLMModelException):
    """Exception raised when trying to create a model with a duplicate name for the same provider."""

    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model {model_name} already exists for this provider")
