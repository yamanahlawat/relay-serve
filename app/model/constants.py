from pydantic import BaseModel, Field


class LLMDefaults(BaseModel):
    """
    Default parameters for LLM requests
    """

    TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=4096, gt=0)


# Initialize defaults
llm_defaults = LLMDefaults()
