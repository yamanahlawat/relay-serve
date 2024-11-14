from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """
    Details for provider-specific errors
    """

    code: str
    message: str
    provider: str
    details: Any | None = None


class ErrorResponseModel(BaseModel):
    """
    Standard error response model for OpenAPI documentation
    """

    detail: str | ErrorDetail
