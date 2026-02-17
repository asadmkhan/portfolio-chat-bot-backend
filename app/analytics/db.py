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
    purge_old_records()


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


def purge_old_records() -> dict[str, int]:
    if not settings.analytics_enabled:
        return {"analytics_events": 0, "feedback": 0}

    db_path = _get_db_path()
    analytics_retention = max(1, int(settings.analytics_retention_days))
    feedback_retention = max(1, int(settings.feedback_retention_days))

    deleted = {"analytics_events": 0, "feedback": 0}
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM analytics_events WHERE created_at < datetime('now', ?)",
            (f"-{analytics_retention} days",),
        )
        deleted["analytics_events"] = int(cur.rowcount or 0)

        cur = conn.execute(
            "DELETE FROM feedback WHERE created_at < datetime('now', ?)",
            (f"-{feedback_retention} days",),
        )
        deleted["feedback"] = int(cur.rowcount or 0)
        conn.commit()

    return deleted


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_summary() -> dict[str, Any]:
    if not settings.analytics_enabled:
        return {"enabled": False}
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) AS total FROM analytics_events")
        total = cur.fetchone()[0]
        cur = conn.execute(
            """
            SELECT COUNT(*) AS total_7d
            FROM analytics_events
            WHERE created_at >= datetime('now', '-7 days')
            """
        )
        total_7d = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) AS feedback_total FROM feedback")
        feedback_total = cur.fetchone()[0]
    return {
        "enabled": True,
        "total": total,
        "total_7d": total_7d,
        "feedback_total": feedback_total,
    }


def get_latest(limit: int = 20) -> list[dict[str, Any]]:
    if not settings.analytics_enabled:
        return []
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT created_at, conversation_id, message_id, language, question, response
            FROM analytics_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [_row_to_dict(cur, row) for row in rows]


def get_top_questions(limit: int = 10) -> list[dict[str, Any]]:
    if not settings.analytics_enabled:
        return []
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT question, COUNT(*) as count
            FROM analytics_events
            WHERE question IS NOT NULL AND question != ''
            GROUP BY question
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [_row_to_dict(cur, row) for row in rows]


def get_feedback(limit: int = 20) -> list[dict[str, Any]]:
    if not settings.analytics_enabled:
        return []
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT created_at, conversation_id, message_id, rating, comment
            FROM feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [_row_to_dict(cur, row) for row in rows]
