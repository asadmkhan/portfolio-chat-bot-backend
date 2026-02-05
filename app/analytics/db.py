from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db_path() -> Path:
    return Path(settings.analytics_db_path)


def init_db() -> None:
    if not settings.analytics_enabled:
        return
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                conversation_id TEXT,
                message_id TEXT,
                language TEXT,
                question TEXT,
                response TEXT,
                k INTEGER,
                use_mmr INTEGER,
                fetch_k INTEGER,
                mmr_lambda REAL,
                sources_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                conversation_id TEXT,
                message_id TEXT,
                rating TEXT NOT NULL,
                comment TEXT
            )
            """
        )
        conn.commit()


def log_chat_event(
    *,
    conversation_id: str | None,
    message_id: str | None,
    language: str,
    question: str,
    response: str,
    k: int,
    use_mmr: bool,
    fetch_k: int,
    mmr_lambda: float,
    sources: list[dict[str, Any]],
) -> None:
    if not settings.analytics_enabled:
        return
    db_path = _get_db_path()
    sources_json = json.dumps(sources, ensure_ascii=False)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analytics_events (
                created_at, conversation_id, message_id, language, question, response,
                k, use_mmr, fetch_k, mmr_lambda, sources_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                conversation_id,
                message_id,
                language,
                question,
                response,
                k,
                1 if use_mmr else 0,
                fetch_k,
                mmr_lambda,
                sources_json,
            ),
        )
        conn.commit()


def log_feedback(
    *,
    conversation_id: str | None,
    message_id: str | None,
    rating: str,
    comment: str | None,
) -> None:
    if not settings.analytics_enabled:
        return
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                created_at, conversation_id, message_id, rating, comment
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                conversation_id,
                message_id,
                rating,
                comment,
            ),
        )
        conn.commit()
