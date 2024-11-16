"""
Core exceptions module for Relay application.
Contains base exceptions that can be extended by other modules.
"""

from typing import Any

from fastapi import HTTPException


class RelayException(HTTPException):
    """
    Base exception for all Relay application exceptions.
    Ensures consistent error response structure across the application.

    Example:
        >>> raise RelayException(
        ...     status_code=500,
        ...     error_code="internal_error",
        ...     message="Internal server error occurred",
        ...     context={"trace_id": "abc123"},
        ... )
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        context: Any | None = None,
        loc: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the exception with error details.

        Args:
            status_code: HTTP status code
            error_code: Machine-readable error code
            message: Human-readable error message
            context: Additional error context
            loc: Error location (e.g., ["body", "field_name"])
            headers: Optional HTTP headers for the response
        """
        super().__init__(
            status_code=status_code,
            detail=[
                {
                    "type": error_code,
                    "loc": loc or ["application"],
                    "message": message,
                    "context": context,
                }
            ],
            headers=headers,
        )
