from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings

_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _get_connection() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is not None:
            return _conn

        db_path = settings.recruiter_share_db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        _conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=5,
            isolation_level=None,
        )
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA synchronous=NORMAL;")
        _conn.execute("PRAGMA busy_timeout=5000;")
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recruiter_share_records (
                share_id TEXT PRIMARY KEY,
                tool_slug TEXT NOT NULL,
                locale TEXT NOT NULL,
                result_payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            """
        )
        _conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recruiter_share_expiry
            ON recruiter_share_records (expires_at);
            """
        )
        return _conn


def purge_expired_recruiter_shares() -> None:
    conn = _get_connection()
    now_iso = _utc_now().isoformat()
    with _conn_lock:
        conn.execute("DELETE FROM recruiter_share_records WHERE expires_at <= ?", (now_iso,))
        conn.commit()


def create_recruiter_share(*, tool_slug: str, locale: str, result_payload: dict[str, Any]) -> tuple[str, datetime]:
    conn = _get_connection()
    purge_expired_recruiter_shares()
    ttl_days = max(1, int(settings.recruiter_share_ttl_days))
    created_at = _utc_now()
    expires_at = created_at + timedelta(days=ttl_days)
    share_id = secrets.token_urlsafe(9)
    payload_json = json.dumps(result_payload, ensure_ascii=False)

    with _conn_lock:
        conn.execute(
            """
            INSERT INTO recruiter_share_records (
                share_id, tool_slug, locale, result_payload_json, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                share_id,
                tool_slug,
                locale,
                payload_json,
                created_at.isoformat(),
                expires_at.isoformat(),
            ),
        )
        conn.commit()
    return share_id, expires_at


def get_recruiter_share(share_id: str) -> dict[str, Any] | None:
    conn = _get_connection()
    purge_expired_recruiter_shares()
    with _conn_lock:
        cur = conn.execute(
            """
            SELECT share_id, tool_slug, locale, result_payload_json, created_at, expires_at
            FROM recruiter_share_records
            WHERE share_id = ?
            """,
            (share_id,),
        )
        row = cur.fetchone()

    if not row:
        return None

    result_payload = json.loads(row[3]) if row[3] else {}
    return {
        "share_id": row[0],
        "tool_slug": row[1],
        "locale": row[2],
        "result_payload": result_payload,
        "created_at": datetime.fromisoformat(row[4]),
        "expires_at": datetime.fromisoformat(row[5]),
    }

