from pydantic import BaseModel, Field
from typing import Any


class ChatMessage(BaseModel):
    role: str = Field(description="system | user | assistant")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool | None = False
    # любые доп поля прокидываем - openai совместимость
    extra: dict[str, Any] | None = None

    class Config:
        extra = "allow"


class CompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    prompt: str
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool | None = False

    class Config:
        extra = "allow"
