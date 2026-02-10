from __future__ import annotations

from app.core.config import settings


def cors_allowed_origins() -> list[str]:
    return list(settings.cors_allowed_origins)


def cors_allow_origin_regex() -> str | None:
    regex = (settings.cors_allow_origin_regex or "").strip()
    return regex or None
