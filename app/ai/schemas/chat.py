from pydantic import BaseModel, Field

from app.chat.constants import llm_defaults


class CompletionParams(BaseModel):
    """
    Parameters for completion generation
    """

    max_tokens: int = Field(default=llm_defaults.MAX_TOKENS, gt=0)
    temperature: float = Field(default=llm_defaults.TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=llm_defaults.TOP_P, ge=0.0, le=1.0)


class TransparencySettings(BaseModel):
    """Settings controlling AI transparency and visibility"""

    show_thinking: bool = Field(default=True, description="Show AI reasoning and thinking processes")
    show_tool_calls: bool = Field(default=True, description="Show when AI uses tools and their arguments")
    show_tool_results: bool = Field(default=True, description="Show results from tool calls")
    show_typing_indicators: bool = Field(default=True, description="Show typing indicators during processing")
    show_part_events: bool = Field(default=False, description="Show detailed part start/delta events for debugging")
