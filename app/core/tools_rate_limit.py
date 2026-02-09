from __future__ import annotations

import os
import sqlite3
import threading
import time

from app.core.config import settings

_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


class ToolsRateLimitExceeded(Exception):
    pass


def _get_connection() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is not None:
            return _conn

        db_path = settings.tools_rate_limit_db_path
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
            CREATE TABLE IF NOT EXISTS tools_rate_limit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_key TEXT NOT NULL,
                route_key TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        _conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tools_rate_limit_lookup
            ON tools_rate_limit_events (client_key, route_key, created_at);
            """
        )
        return _conn


def enforce_tools_rate_limit(client_key: str, route_key: str, limit: int, window_seconds: int = 60) -> None:
    now = time.time()
    cutoff = now - window_seconds
    conn = _get_connection()

    with _conn_lock:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute("DELETE FROM tools_rate_limit_events WHERE created_at < ?", (cutoff,))
            cursor.execute(
                """
                SELECT COUNT(1)
                FROM tools_rate_limit_events
                WHERE client_key = ? AND route_key = ? AND created_at >= ?
                """,
                (client_key, route_key, cutoff),
            )
            count = int(cursor.fetchone()[0] or 0)
            if count >= limit:
                conn.rollback()
                raise ToolsRateLimitExceeded

            cursor.execute(
                """
                INSERT INTO tools_rate_limit_events (client_key, route_key, created_at)
                VALUES (?, ?, ?)
                """,
                (client_key, route_key, now),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def clear_tools_rate_limit_events() -> None:
    conn = _get_connection()
    with _conn_lock:
        conn.execute("DELETE FROM tools_rate_limit_events")
        conn.commit()
