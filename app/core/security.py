from __future__ import annotations

from fastapi import HTTPException, status

from app.core.config import settings


def _normalize_lang(lang: str | None) -> str:
    if not lang:
        return "en"
    return lang.split(",")[0].strip().lower()


def _auth_error_message(lang: str | None) -> str:
    key = _normalize_lang(lang)
    messages = {
        "en": "Please provide a valid API key to use the chatbot.",
        "de": "Bitte gib einen gültigen API‑Schlüssel an, um den Chatbot zu nutzen.",
        "fr": "Veuillez fournir une clé API valide pour utiliser le chatbot.",
        "es": "Por favor, proporciona una clave API válida para usar el chatbot.",
        "it": "Per favore, fornisci una chiave API valida per usare il chatbot.",
    }
    return messages.get(key, messages["en"])


def check_api_key(x_api_key: str | None, lang: str | None = None) -> None:
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_auth_error_message(lang),
        )
