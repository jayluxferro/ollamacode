"""
Session persistence: SQLite backend for chat sessions with optional FTS5 search.

Sessions are stored in ~/.ollamacode/sessions.db. Schema: sessions (id, title, created_at, updated_at, message_count);
messages (session_id, seq, role, content). FTS5 virtual table for search when available.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
            workspace_root TEXT NOT NULL DEFAULT '',
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
    # Add columns for older DBs (best-effort migrations).
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "workspace_root" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN workspace_root TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(
            "Session DB migration failed (adding workspace_root column): %s", exc
        )
    # Restrict DB file to owner-only so session history isn't world-readable.
    try:
        _DB_PATH.chmod(0o600)
    except OSError:
        pass


def create_session(title: str = "", workspace_root: str | None = None) -> str:
    """Create a new session and return its id."""
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, title, workspace_root, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, 0)",
            (sid, (title or "").strip()[:500], workspace_root or "", now, now),
        )
        conn.commit()
    return sid


def save_session(
    session_id: str,
    title: str | None,
    message_history: list[dict[str, Any]],
    workspace_root: str | None = None,
) -> None:
    """Save or update a session: title and full message list (replaces existing messages)."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        conn.execute("BEGIN")
        try:
            cur = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
            if cur.fetchone():
                conn.execute(
                    "UPDATE sessions SET title = ?, workspace_root = ?, updated_at = ?, message_count = ? WHERE id = ?",
                    (
                        (title or "").strip()[:500],
                        workspace_root or "",
                        now,
                        len(message_history),
                        session_id,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO sessions (id, title, workspace_root, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        (title or "").strip()[:500],
                        workspace_root or "",
                        now,
                        now,
                        len(message_history),
                    ),
                )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for seq, msg in enumerate(message_history):
                if not isinstance(msg, dict) or "role" not in msg:
                    logger.warning(
                        "Skipping invalid message at index %d in session %s: missing 'role' field",
                        seq,
                        session_id,
                    )
                    continue
                role = (msg.get("role") or "user").strip()[:64]
                content = msg.get("content") or ""
                if len(content) > _MAX_MESSAGE_LEN:
                    content = content[:_MAX_MESSAGE_LEN] + "\n... [truncated]"
                conn.execute(
                    "INSERT INTO messages (session_id, seq, role, content) VALUES (?, ?, ?, ?)",
                    (session_id, seq, role, content),
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


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
    """Return session row as dict (id, title, workspace_root, created_at, updated_at, message_count) or None."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT id, title, workspace_root, created_at, updated_at, message_count FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "workspace_root": row[2],
        "created_at": row[3],
        "updated_at": row[4],
        "message_count": row[5],
    }


def list_sessions(
    limit: int = 50, workspace_root: str | None = None
) -> list[dict[str, Any]]:
    """List recent sessions (id, title, workspace_root, created_at, updated_at, message_count)."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        if workspace_root:
            cur = conn.execute(
                "SELECT id, title, workspace_root, created_at, updated_at, message_count FROM sessions WHERE workspace_root = ? ORDER BY updated_at DESC LIMIT ?",
                (workspace_root, limit),
            )
        else:
            cur = conn.execute(
                "SELECT id, title, workspace_root, created_at, updated_at, message_count FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "workspace_root": r[2],
            "created_at": r[3],
            "updated_at": r[4],
            "message_count": r[5],
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
            SELECT DISTINCT s.id, s.title, s.workspace_root, s.created_at, s.updated_at, s.message_count
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
            "workspace_root": r[2],
            "created_at": r[3],
            "updated_at": r[4],
            "message_count": r[5],
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
    workspace_root = info.get("workspace_root") if info else None
    new_id = create_session(title=new_title, workspace_root=workspace_root)
    save_session(new_id, new_title, messages, workspace_root=workspace_root)
    return new_id


def fork_session(
    session_id: str,
    message_index: int,
    title: str | None = None,
) -> str | None:
    """Fork a session at a specific message index.

    Copies messages [0, message_index] (inclusive) into a new session.
    Returns the new session id, or None if the source session is not found
    or message_index is out of range.
    """
    messages = load_session(session_id)
    if messages is None:
        return None
    if message_index < 0 or message_index >= len(messages):
        logger.warning(
            "fork_session: message_index %d out of range (session has %d messages)",
            message_index,
            len(messages),
        )
        return None
    forked_messages = messages[: message_index + 1]
    info = get_session_info(session_id)
    src_title = info.get("title", "") if info else ""
    new_title = (title or "").strip() or f"Fork of {src_title}" if src_title else "Fork"
    workspace_root = info.get("workspace_root") if info else None
    new_id = create_session(title=new_title, workspace_root=workspace_root)
    save_session(new_id, new_title, forked_messages, workspace_root=workspace_root)
    logger.info(
        "Forked session %s at message %d -> %s (%d messages)",
        session_id,
        message_index,
        new_id,
        len(forked_messages),
    )
    return new_id


def get_latest_session(workspace_root: str) -> dict[str, Any] | None:
    """Return the most recently updated session for a workspace."""
    rows = list_sessions(limit=1, workspace_root=workspace_root)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Session sharing: export / import
# ---------------------------------------------------------------------------

_EXPORT_VERSION = 1


def export_session(session_id: str) -> str | None:
    """Export a session as a JSON string for sharing.

    Returns a JSON string containing session metadata and messages,
    or None if the session is not found.
    """
    info = get_session_info(session_id)
    if info is None:
        return None
    messages = load_session(session_id)
    if messages is None:
        return None

    export_data = {
        "version": _EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "session": {
            "title": info.get("title", ""),
            "workspace_root": info.get("workspace_root", ""),
            "created_at": info.get("created_at", ""),
            "updated_at": info.get("updated_at", ""),
        },
        "messages": messages,
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def import_session(json_str: str, title: str | None = None) -> str:
    """Import a session from a JSON string.

    Returns the new session ID.
    Raises ValueError if the JSON is invalid or missing required fields.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at top level")

    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Missing or invalid 'messages' list in export data")

    session_meta = data.get("session") or {}
    import_title = (
        (title or "").strip()
        or (session_meta.get("title") or "").strip()
        or "Imported session"
    )
    workspace_root = session_meta.get("workspace_root") or ""

    new_id = create_session(title=import_title, workspace_root=workspace_root)
    save_session(new_id, import_title, messages, workspace_root=workspace_root)

    logger.info(
        "Imported session %s with %d messages (title: %s)",
        new_id,
        len(messages),
        import_title,
    )
    return new_id
