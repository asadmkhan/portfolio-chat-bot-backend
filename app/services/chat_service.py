from typing import AsyncGenerator

from app.utils.sse import sse
from app.core import events

from app.rag.lang import detect_lang
from app.rag.retriever import FaissRetriever
from app.rag.prompt import build_rag_messages

from app.ai.factory import get_ai_client


async def stream_chat(message: str) -> AsyncGenerator[str, None]:
    try:
        yield sse(events.TRACE, "Thinking...")

        user_message = (message or "").strip()
        if not user_message:
            yield sse(events.CHUNK, "Please type a message.")
            yield sse(events.DONE, "[DONE]")
            return

        lang = detect_lang(user_message)

        retriever = FaissRetriever.get(lang)
        chunks = retriever.search(user_message, k=5)

        messages = build_rag_messages(user_message, chunks, lang)

        ai = get_ai_client()
        async for token in ai.stream(messages):
            yield sse(events.CHUNK, token)

        yield sse(events.DONE, "[DONE]")

    except Exception as ex:
        yield sse(events.ERROR, str(ex))
        yield sse(events.DONE, "[DONE]")
