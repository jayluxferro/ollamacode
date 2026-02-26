"""
Session persistence: SQLite backend for chat sessions with optional FTS5 search.

Sessions are stored in ~/.ollamacode/sessions.db. Schema: sessions (id, title, created_at, updated_at, message_count);
messages (session_id, seq, role, content). FTS5 virtual table for search when available.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path.home() / ".ollamacode" / "sessions.db"
_MAX_MESSAGE_LEN = 500_000


def _db_path() -> Path:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            PRIMARY KEY (session_id, seq),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
    """)
    conn.commit()
    # Restrict DB file to owner-only so session history isn't world-readable.
    try:
        _DB_PATH.chmod(0o600)
    except OSError:
        pass


def create_session(title: str = "") -> str:
    """Create a new session and return its id."""
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, 0)",
            (sid, (title or "").strip()[:500], now, now),
        )
        conn.commit()
    return sid


def save_session(
    session_id: str, title: str | None, message_history: list[dict[str, Any]]
) -> None:
    """Save or update a session: title and full message list (replaces existing messages)."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        if cur.fetchone():
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ?, message_count = ? WHERE id = ?",
                ((title or "").strip()[:500], now, len(message_history), session_id),
            )
        else:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    (title or "").strip()[:500],
                    now,
                    now,
                    len(message_history),
                ),
            )
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        for seq, msg in enumerate(message_history):
            role = (msg.get("role") or "user").strip()[:64]
            content = msg.get("content") or ""
            if len(content) > _MAX_MESSAGE_LEN:
                content = content[:_MAX_MESSAGE_LEN] + "\n... [truncated]"
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content) VALUES (?, ?, ?, ?)",
                (session_id, seq, role, content),
            )
        conn.commit()


def load_session(session_id: str) -> list[dict[str, Any]] | None:
    """Load message_history for a session, or None if not found."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        )
        rows = cur.fetchall()
        if not rows:
            # Distinguish "session exists but empty" from "session not found" in one transaction.
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                return None
            return []
    return [{"role": r, "content": c} for r, c in rows]


def get_session_info(session_id: str) -> dict[str, Any] | None:
    """Return session row as dict (id, title, created_at, updated_at, message_count) or None."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT id, title, created_at, updated_at, message_count FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "created_at": row[2],
        "updated_at": row[3],
        "message_count": row[4],
    }


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List recent sessions (id, title, created_at, updated_at, message_count)."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT id, title, created_at, updated_at, message_count FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "message_count": r[4],
        }
        for r in rows
    ]


def search_sessions(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search sessions by message content. Uses LIKE on messages if FTS5 not available."""
    q = (query or "").strip()
    if not q:
        return list_sessions(limit=limit)
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        # Simple LIKE search across messages and session title
        cur = conn.execute(
            """
            SELECT DISTINCT s.id, s.title, s.created_at, s.updated_at, s.message_count
            FROM sessions s
            JOIN messages m ON m.session_id = s.id
            WHERE m.content LIKE ? OR s.title LIKE ?
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (f"%{q}%", f"%{q}%", limit),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "message_count": r[4],
        }
        for r in rows
    ]


def branch_session(session_id: str, title: str | None = None) -> str | None:
    """Copy current session's messages to a new session; return new id or None if source not found."""
    messages = load_session(session_id)
    if messages is None:
        return None
    info = get_session_info(session_id)
    new_title = (title or "").strip() or (
        f"Branch of {info['title']}" if info and info.get("title") else "Branch"
    )
    new_id = create_session(title=new_title)
    save_session(new_id, new_title, messages)
    return new_id
