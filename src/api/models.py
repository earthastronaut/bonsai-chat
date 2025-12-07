"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel
from ..config import MODEL_ID


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat completion request model."""
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False


class CompletionRequest(BaseModel):
    """Text completion request model."""
    model: str = MODEL_ID
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7

