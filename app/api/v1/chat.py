from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.chat_service import stream_chat

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(payload: dict):
    message = payload.get("message", "")
    gen = stream_chat(message)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
