from typing import List, Dict
from app.ai.types import ChatMessage


def build_context(chunks, max_chars_per_chunk=900):
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source", "unknown")
        chunk_text = (c.get("text") or "").strip()
        if len(chunk_text) > max_chars_per_chunk:
            chunk_text = chunk_text[:max_chars_per_chunk] + "..."
        lines.append(f"[{i}] source: {src}\n{chunk_text}\n")
    return "\n".join(lines).strip()


def build_rag_messages(
    user_question: str,
    chunks: List[Dict],
    lang: str,
    max_chars_per_chunk: int | None = None,
) -> list[ChatMessage]:
    context = build_context(
        chunks,
        max_chars_per_chunk=max_chars_per_chunk or 900,
    )

    if lang == "de":
        system = (
            "Du bist ein Portfolio-Assistent fuer Asad Khan. "
            "Antworte nur mit Informationen aus dem bereitgestellten Kontext. "
            "Wenn etwas nicht im Kontext steht, sag ehrlich: 'Das steht nicht in meinen Unterlagen.' "
            "Antworte als klarer Fliesstext fuer das Streaming. "
            "Haenge am ENDE eine JSON-Struktur in einem <json>...</json>-Block an. "
            "Schema: {\"summary\": string, \"items\": [string], \"details\": [{\"title\": string, \"bullets\": [string]}], \"notes\": string}. "
            "Wenn etwas fehlt, nutze leere Liste oder leeren String."
        )
        user = f"FRAGE:\n{user_question}\n\nKONTEXT:\n{context}\n\nANTWORT:"
    else:
        system = (
            "You are a portfolio assistant for Asad Khan. "
            "Answer only using the provided context. "
            "If the answer is not in the context, say: 'I don't have that in my documents.' "
            "Respond as clear plain text for streaming. "
            "Append a JSON block at the END inside <json>...</json>. "
            "Schema: {\"summary\": string, \"items\": [string], \"details\": [{\"title\": string, \"bullets\": [string]}], \"notes\": string}. "
            "If something is missing, use empty list or empty string."
        )
        user = f"QUESTION:\n{user_question}\n\nCONTEXT:\n{context}\n\nANSWER:"

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
