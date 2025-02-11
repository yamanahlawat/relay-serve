from typing import Any, Literal

from pydantic import BaseModel


class AnthropicToolInput(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any]
    required: list[str] | None = None


class AnthropicTool(BaseModel):
    name: str
    description: str
    parameters: AnthropicToolInput
