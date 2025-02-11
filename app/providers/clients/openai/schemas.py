from typing import Any, Literal

from pydantic import BaseModel


class OpenAIFunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any]
    required: list[str] | None = None
    additionalProperties: bool = False


class OpenAIFunction(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParameters
    strict: bool = False


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunction
