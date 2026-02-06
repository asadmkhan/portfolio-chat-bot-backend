from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_env_bool(name: str, default: bool) -> bool:
    raw = _get_env(name, None)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    raw = _get_env(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    rate_limit: str
    rate_limit_enabled: bool
    log_level: str
    sentry_dsn: str | None
    log_message_max_chars: int
    mmr_use: bool
    mmr_fetch_k: int
    mmr_lambda: float
    analytics_enabled: bool
    analytics_db_path: str
    min_chunk_score: float
    contact_email: str
    contact_linkedin: str


settings = Settings(
    api_key=_get_env("API_KEY"),
    rate_limit=_get_env("RATE_LIMIT", "60/minute") or "60/minute",
    rate_limit_enabled=_get_env_bool("RATE_LIMIT_ENABLED", True),
    log_level=_get_env("LOG_LEVEL", "INFO") or "INFO",
    sentry_dsn=_get_env("SENTRY_DSN"),
    log_message_max_chars=_get_env_int("LOG_MESSAGE_MAX_CHARS", 800),
    mmr_use=_get_env_bool("MMR_USE", True),
    mmr_fetch_k=_get_env_int("MMR_FETCH_K", 10),
    mmr_lambda=float(_get_env("MMR_LAMBDA", "0.7") or "0.7"),
    analytics_enabled=_get_env_bool("ANALYTICS_ENABLED", True),
    analytics_db_path=_get_env("ANALYTICS_DB_PATH", "data/analytics.db") or "data/analytics.db",
    min_chunk_score=float(_get_env("MIN_CHUNK_SCORE", "0.25") or "0.25"),
    contact_email=_get_env("CONTACT_EMAIL", "contact@codedbyasad.com") or "contact@codedbyasad.com",
    contact_linkedin=_get_env("CONTACT_LINKEDIN", "https://www.linkedin.com/in/asadmkhan") or "https://www.linkedin.com/in/asadmkhan",
)
