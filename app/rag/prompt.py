from typing import List, Dict
from app.ai.types import ChatMessage


def build_context(chunks, max_chars_per_chunk=900):
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source", "unknown")
        text = (c.get("text") or "").strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "…"
        lines.append(f"[{i}] source: {src}\n{text}\n")
    return "\n".join(lines).strip()


def build_rag_messages(
    user_question: str, chunks: List[Dict], lang: str
) -> list[ChatMessage]:
    context = build_context(chunks)

    if lang == "de":
        system = (
            "Du bist ein Portfolio‑Assistent für Asad Khan. "
            "Antworte nur mit Informationen aus dem bereitgestellten Kontext. "
            "Wenn etwas nicht im Kontext steht, sag ehrlich: 'Das steht nicht in meinen Unterlagen.' "
        )
        user = f"FRAGE:\n{user_question}\n\n" f"KONTEXT:\n{context}\n\n" "ANTWORT:"
    else:
        system = (
            "You are a portfolio assistant for Asad Khan. "
            "Answer only using the provided context. "
            "If the answer is not in the context, say: 'I don't have that in my documents.' "
            "Use short paragraphs and normal spacing."
        )
        user = f"QUESTION:\n{user_question}\n\n" f"CONTEXT:\n{context}\n\n" "ANSWER:"

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
