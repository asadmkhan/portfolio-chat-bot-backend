from typing import AsyncGenerator
import json
import logging
import time

from app.utils.sse import sse
from app.core import events
from app.core.config import settings

from app.rag.lang import detect_lang
from app.rag.retriever import FaissRetriever
from app.rag.prompt import build_rag_messages

from app.ai.factory import get_ai_client

logger = logging.getLogger("app.chat")


async def stream_chat(
    message: str,
    *,
    language: str | None = None,
    k: int | None = None,
    max_chars_per_chunk: int | None = None,
    conversation_id: str | None = None,
    compress_context: bool | None = None,
    include_citations: bool | None = None,
) -> AsyncGenerator[str, None]:
    started_at = time.perf_counter()
    try:
        yield sse(events.TRACE, "Thinking...")

        user_message = (message or "").strip()
        if not user_message:
            yield sse(events.CHUNK, "Please type a message.")
            yield sse(events.DONE, "[DONE]")
            return

        lang = (language or "").strip().lower() or detect_lang(user_message)

        retriever = FaissRetriever.get(lang)
        top_k = k or 5
        chunks = retriever.search(user_message, k=top_k)

        messages = build_rag_messages(
            user_message,
            chunks,
            lang,
            max_chars_per_chunk=max_chars_per_chunk,
        )

        logger.info(
            json.dumps(
                {
                    "event": "chat_request",
                    "lang": lang,
                    "k": top_k,
                    "conversation_id": conversation_id,
                    "compress_context": compress_context,
                    "include_citations": include_citations,
                    "message": user_message[: settings.log_message_max_chars],
                    "chunk_ids": [c.get("id") for c in chunks],
                }
            )
        )

        ai = get_ai_client()
        async for token in ai.stream(messages):
            yield sse(events.CHUNK, token)

        yield sse(events.DONE, "[DONE]")

    except Exception as ex:
        logger.exception(
            json.dumps(
                {
                    "event": "chat_error",
                    "error": str(ex),
                    "duration_ms": int((time.perf_counter() - started_at) * 1000),
                }
            )
        )
        yield sse(events.ERROR, str(ex))
        yield sse(events.DONE, "[DONE]")
    else:
        logger.info(
            json.dumps(
                {
                    "event": "chat_complete",
                    "duration_ms": int((time.perf_counter() - started_at) * 1000),
                }
            )
        )
