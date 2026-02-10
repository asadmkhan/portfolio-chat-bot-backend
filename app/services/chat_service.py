from typing import AsyncGenerator
import hashlib
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
from app.analytics.db import log_chat_event

logger = logging.getLogger("app.chat")
_conversations: dict[str, list[dict[str, str]]] = {}
_max_history = 6


def _short_hash(value: str | None) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _strip_json_block(text: str) -> str:
    start = text.find("<json>")
    end = text.find("</json>")
    if start == -1:
        return text.strip()
    if end == -1:
        return text[:start].strip()
    return (text[:start] + text[end + 7 :]).strip()


def _get_history(conversation_id: str | None) -> list[dict[str, str]]:
    if not conversation_id:
        return []
    history = _conversations.get(conversation_id, [])
    return history[-_max_history :]


def _append_history(conversation_id: str | None, role: str, content: str) -> None:
    if not conversation_id:
        return
    history = _conversations.setdefault(conversation_id, [])
    history.append({"role": role, "content": content})
    if len(history) > _max_history * 2:
        del history[:-_max_history * 2]


async def stream_chat(
    message: str,
    *,
    language: str | None = None,
    k: int | None = None,
    max_chars_per_chunk: int | None = None,
    conversation_id: str | None = None,
    compress_context: bool | None = None,
    include_citations: bool | None = None,
    use_mmr: bool | None = None,
    fetch_k: int | None = None,
    mmr_lambda: float | None = None,
    message_id: str | None = None,
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
        resolved_use_mmr = settings.mmr_use if use_mmr is None else use_mmr
        resolved_fetch_k = (
            settings.mmr_fetch_k if fetch_k is None else fetch_k
        ) or max(10, top_k * 3)
        resolved_mmr_lambda = settings.mmr_lambda if mmr_lambda is None else mmr_lambda
        chunks = retriever.search(
            user_message,
            k=top_k,
            use_mmr=resolved_use_mmr,
            fetch_k=resolved_fetch_k,
            mmr_lambda=resolved_mmr_lambda,
        )

        history = _get_history(conversation_id)
        messages = build_rag_messages(
            user_message,
            chunks,
            lang,
            max_chars_per_chunk=max_chars_per_chunk,
            history=history,
        )

        resolved_include_citations = True if include_citations is None else include_citations
        max_score = max((c.get("score", 0.0) or 0.0) for c in chunks) if chunks else 0.0
        low_confidence = len(chunks) == 0 or max_score < settings.min_chunk_score
        if resolved_include_citations:
            sources = [
                {"id": c.get("id"), "source": c.get("source"), "score": c.get("score")}
                for c in chunks
            ]
            yield sse(events.SOURCES, json.dumps(sources))
        if low_confidence:
            yield sse(
                events.CTA,
                json.dumps(
                    {
                        "email": settings.contact_email,
                        "linkedin": settings.contact_linkedin,
                        "reason": "low_confidence",
                        "message": (
                            "I couldn't find this in my documents. "
                            "Please reach out via email or LinkedIn and I’ll respond quickly."
                        ),
                    }
                ),
            )

        logger.info(
            json.dumps(
                {
                    "event": "chat_request",
                    "lang": lang,
                    "k": top_k,
                    "conversation_hash": _short_hash(conversation_id),
                    "message_id_hash": _short_hash(message_id),
                    "compress_context": compress_context,
                    "include_citations": resolved_include_citations,
                    "use_mmr": resolved_use_mmr,
                    "fetch_k": resolved_fetch_k,
                    "mmr_lambda": resolved_mmr_lambda,
                    "max_score": max_score,
                    "message_len": len(user_message),
                    "message_hash": _short_hash(user_message),
                    "chunk_ids": [c.get("id") for c in chunks],
                }
            )
        )

        _append_history(conversation_id, "user", user_message)

        ai = get_ai_client()
        assistant_text = ""
        async for token in ai.stream(messages):
            assistant_text += token
            yield sse(events.CHUNK, token)

        yield sse(events.DONE, "[DONE]")
        _append_history(conversation_id, "assistant", _strip_json_block(assistant_text))
        log_chat_event(
            conversation_id=conversation_id,
            message_id=message_id,
            language=lang,
            question=user_message,
            response=_strip_json_block(assistant_text),
            k=top_k,
            use_mmr=resolved_use_mmr,
            fetch_k=resolved_fetch_k,
            mmr_lambda=resolved_mmr_lambda,
            sources=[
                {"id": c.get("id"), "source": c.get("source"), "score": c.get("score")}
                for c in chunks
            ],
        )

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
