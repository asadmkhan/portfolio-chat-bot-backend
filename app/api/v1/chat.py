from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.rate_limit import rate_limit
from app.core.security import check_api_key
from app.services.chat_service import stream_chat

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = ""
    language: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)
    max_chars_per_chunk: int | None = Field(default=None, ge=200, le=2000)
    conversation_id: str | None = None
    compress_context: bool | None = None
    include_citations: bool | None = None


@router.post("/chat/stream")
@rate_limit()
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _ = request
    check_api_key(x_api_key, payload.language)
    gen = stream_chat(
        payload.message,
        language=payload.language,
        k=payload.k,
        max_chars_per_chunk=payload.max_chars_per_chunk,
        conversation_id=payload.conversation_id,
        compress_context=payload.compress_context,
        include_citations=payload.include_citations,
    )

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
