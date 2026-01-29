from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    conversation_id: str | None = None
