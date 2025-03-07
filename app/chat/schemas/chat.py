from pydantic import BaseModel, Field

from app.chat.constants import llm_defaults


class CompletionParams(BaseModel):
    """
    Parameters for completion generation
    """

    max_tokens: int = Field(default=llm_defaults.MAX_TOKENS, gt=0)
    temperature: float = Field(default=llm_defaults.TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=llm_defaults.TOP_P, ge=0.0, le=1.0)
