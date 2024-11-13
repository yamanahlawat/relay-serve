from pydantic import BaseModel, Field


class ChatUsage(BaseModel):
    """
    Schema for token usage and costs
    """

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float = Field(description="Total cost (input + output)")
