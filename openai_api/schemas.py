from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


SUPPORTED_MODEL = "gemma-3-270m-it"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_768)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=SUPPORTED_MODEL)
    messages: List[ChatMessage] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=128, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value != SUPPORTED_MODEL:
            raise ValueError(f"Only '{SUPPORTED_MODEL}' is supported")
        return value


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
