"""
Provider-specific exceptions for handling LLM provider errors.
These exceptions ensure consistent error handling across different providers.
"""

from typing import Any

from fastapi import status

from app.api.exceptions import RelayException
from app.providers.constants import ProviderErrorCode, ProviderType


class ProviderException(RelayException):
    """
    Base exception for provider-related errors.
    Provides consistent error structure for all provider operations.

    Example:
        >>> raise ProviderException(
        ...     status_code=503,
        ...     error_code=ProviderErrorCode.CONNECTION,
        ...     message="Provider service unavailable",
        ...     provider=ProviderType.ANTHROPIC,
        ... )
    """

    def __init__(
        self,
        status_code: int,
        error_code: ProviderErrorCode,
        message: str,
        provider: ProviderType,
        error: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize provider exception with error details.

        Args:
            status_code: HTTP status code
            error_code: Provider-specific error code
            message: Human-readable error message
            provider: Type of the provider
            error: Provider-specific error details
            headers: Optional HTTP headers for the response
        """
        super().__init__(
            status_code=status_code,
            error_code=error_code.value,
            message=message,
            loc=["provider"],
            context={
                "provider": provider,
                "error": error,
            },
            headers=headers,
        )


class ProviderConnectionError(ProviderException):
    """
    Exception raised when provider connection fails.
    This includes network errors, DNS issues, and service unavailability.

    Example:
        >>> raise ProviderConnectionError(
        ...     provider=ProviderType.ANTHROPIC,
        ...     error="Connection timed out after 30s"
        ... )
    """

    def __init__(
        self,
        provider: ProviderType,
        message: str = "Failed to connect to provider API",
        error: Any | None = None,
    ) -> None:
        """
        Initialize connection error.

        Args:
            provider: Type of the provider
            message: Human-readable error message
            error: Connection-specific error details
        """
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ProviderErrorCode.CONNECTION,
            message=message,
            provider=provider,
            error=error,
        )


class ProviderRateLimitError(ProviderException):
    """
    Exception raised when provider rate limits are exceeded.
    This includes both token rate limits and request rate limits.

    Example:
        >>> raise ProviderRateLimitError(
        ...     provider=ProviderType.ANTHROPIC,
        ...     error="Exceeded 10 requests per minute limit"
        ... )
    """

    def __init__(
        self,
        provider: ProviderType,
        message: str = "Rate limit exceeded",
        error: Any | None = None,
    ) -> None:
        """
        Initialize rate limit error.

        Args:
            provider: Type of the provider
            message: Human-readable error message
            error: Rate limit specific error details
        """
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code=ProviderErrorCode.RATE_LIMIT,
            message=message,
            provider=provider,
            error=error,
        )


class ProviderAPIError(ProviderException):
    """
    Exception raised for general provider API errors.
    This includes invalid requests, authentication errors, and internal provider errors.

    Example:
        >>> raise ProviderAPIError(
        ...     provider=ProviderType.ANTHROPIC,
        ...     status_code=400,
        ...     error="Invalid model specified"
        ... )
    """

    def __init__(
        self,
        provider: ProviderType,
        status_code: int,
        message: str = "Provider API error",
        error: Any | None = None,
    ) -> None:
        """
        Initialize API error.

        Args:
            provider: Type of the provider
            status_code: HTTP status code from provider
            message: Human-readable error message
            error: API-specific error details
        """
        super().__init__(
            status_code=status_code,
            error_code=ProviderErrorCode.API,
            message=message,
            provider=provider,
            error=error,
        )


class ProviderConfigurationError(ProviderException):
    """
    Exception raised for provider configuration errors.
    This includes invalid API keys, missing credentials, and invalid settings.

    Example:
        >>> raise ProviderConfigurationError(
        ...     provider=ProviderType.ANTHROPIC,
        ...     error="Invalid API key format"
        ... )
    """

    def __init__(
        self,
        provider: ProviderType,
        message: str = "Provider configuration error",
        error: Any | None = None,
    ) -> None:
        """
        Initialize configuration error.

        Args:
            provider: Type of the provider
            message: Human-readable error message
            error: Configuration-specific error details
        """
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ProviderErrorCode.CONFIGURATION,
            message=message,
            provider=provider,
            error=error,
        )
