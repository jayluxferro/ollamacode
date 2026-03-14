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
_TODO_CONTENT_MAX_LEN = 500
_TODO_STATUS_VALUES = {"pending", "in_progress", "completed", "cancelled"}
_TODO_PRIORITY_VALUES = {"high", "medium", "low"}


def _db_path() -> Path:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            workspace_root TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT NOT NULL DEFAULT '',
            owner TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'owner',
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
        CREATE TABLE IF NOT EXISTS session_todos (
            session_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            priority TEXT NOT NULL DEFAULT 'medium',
            PRIMARY KEY (session_id, position),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_session_todos_session ON session_todos(session_id);
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
        if "parent_session_id" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()
        if "owner" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN owner TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()
        if "role" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN role TEXT NOT NULL DEFAULT 'owner'"
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning("Session DB migration failed (adding sessions columns): %s", exc)
    # Restrict DB file to owner-only so session history isn't world-readable.
    try:
        _DB_PATH.chmod(0o600)
    except OSError:
        pass


def _ensure_session_row(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    title: str = "",
    workspace_root: str = "",
    parent_session_id: str = "",
    owner: str = "",
    role: str = "owner",
) -> None:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is not None:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO sessions (id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (
            session_id,
            title.strip()[:500],
            workspace_root,
            parent_session_id,
            owner,
            role,
            now,
            now,
        ),
    )


def _normalize_todo(todo: Any) -> dict[str, str] | None:
    if not isinstance(todo, dict):
        return None
    content = str(todo.get("content") or todo.get("text") or "").strip()
    if not content:
        return None
    status = (
        str(todo.get("status") or ("completed" if todo.get("done") else "pending"))
        .strip()
        .lower()
    )
    if status not in _TODO_STATUS_VALUES:
        status = "pending"
    priority = str(todo.get("priority") or "medium").strip().lower()
    if priority not in _TODO_PRIORITY_VALUES:
        priority = "medium"
    return {
        "content": content[:_TODO_CONTENT_MAX_LEN],
        "status": status,
        "priority": priority,
    }


def create_session(
    title: str = "",
    workspace_root: str | None = None,
    parent_session_id: str | None = None,
    owner: str | None = None,
    role: str = "owner",
) -> str:
    """Create a new session and return its id."""
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (
                sid,
                (title or "").strip()[:500],
                workspace_root or "",
                parent_session_id or "",
                owner or "",
                role,
                now,
                now,
            ),
        )
        conn.commit()
    try:
        from .control_plane import publish_event

        publish_event(
            "session.created",
            {
                "session_id": sid,
                "title": (title or "").strip()[:500],
                "workspace_root": workspace_root or "",
                "parent_session_id": parent_session_id or "",
                "owner": owner or "",
                "role": role,
            },
        )
    except Exception:
        pass
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
                    "INSERT INTO sessions (id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        (title or "").strip()[:500],
                        workspace_root or "",
                        "",
                        "",
                        "owner",
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
    try:
        from .control_plane import publish_event

        publish_event(
            "session.updated",
            {
                "session_id": session_id,
                "title": (title or "").strip()[:500],
                "workspace_root": workspace_root or "",
                "message_count": len(message_history),
            },
        )
    except Exception:
        pass


def update_session(
    session_id: str,
    *,
    title: str | None = None,
    workspace_root: str | None = None,
    owner: str | None = None,
    role: str | None = None,
) -> dict[str, Any] | None:
    """Update session metadata and return the updated row."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        row = conn.execute(
            "SELECT title, workspace_root, owner, role FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        new_title = (title if title is not None else row[0] or "").strip()[:500]
        new_workspace = (
            workspace_root if workspace_root is not None else row[1] or ""
        ).strip()
        new_owner = (owner if owner is not None else row[2] or "").strip()
        new_role = (role if role is not None else row[3] or "owner").strip() or "owner"
        conn.execute(
            "UPDATE sessions SET title = ?, workspace_root = ?, owner = ?, role = ?, updated_at = ? WHERE id = ?",
            (new_title, new_workspace, new_owner, new_role, now, session_id),
        )
        conn.commit()
    info = get_session_info(session_id)
    if info is not None:
        try:
            from .control_plane import publish_event

            publish_event(
                "session.updated",
                {
                    "session_id": session_id,
                    "title": info.get("title", ""),
                    "workspace_root": info.get("workspace_root", ""),
                    "message_count": info.get("message_count", 0),
                },
            )
        except Exception:
            pass
    return info


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


def save_session_todos(session_id: str, todos: list[dict[str, Any]]) -> None:
    """Save the ordered TODO list for a session."""
    normalized = []
    for todo in todos:
        item = _normalize_todo(todo)
        if item is not None:
            normalized.append(item)

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        conn.execute("BEGIN")
        try:
            _ensure_session_row(conn, session_id)
            conn.execute(
                "DELETE FROM session_todos WHERE session_id = ?", (session_id,)
            )
            for position, todo in enumerate(normalized):
                conn.execute(
                    "INSERT INTO session_todos (session_id, position, content, status, priority) VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        position,
                        todo["content"],
                        todo["status"],
                        todo["priority"],
                    ),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    try:
        from .control_plane import publish_event

        publish_event(
            "session.todos.updated",
            {
                "session_id": session_id,
                "todo_count": len(normalized),
            },
        )
    except Exception:
        pass


def load_session_todos(session_id: str) -> list[dict[str, str]] | None:
    """Load the ordered TODO list for a session."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT content, status, priority FROM session_todos WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
        rows = cur.fetchall()
        if not rows:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                return None
            return []
    return [
        {"content": content, "status": status, "priority": priority}
        for content, status, priority in rows
    ]


def get_session_info(session_id: str) -> dict[str, Any] | None:
    """Return session row as dict (id, title, workspace_root, created_at, updated_at, message_count) or None."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "workspace_root": row[2],
        "parent_session_id": row[3] or "",
        "owner": row[4] or "",
        "role": row[5] or "owner",
        "created_at": row[6],
        "updated_at": row[7],
        "message_count": row[8],
    }


def list_sessions(
    limit: int = 50, workspace_root: str | None = None
) -> list[dict[str, Any]]:
    """List recent sessions (id, title, workspace_root, created_at, updated_at, message_count)."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        if workspace_root:
            cur = conn.execute(
                "SELECT id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count FROM sessions WHERE workspace_root = ? ORDER BY updated_at DESC LIMIT ?",
                (workspace_root, limit),
            )
        else:
            cur = conn.execute(
                "SELECT id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "workspace_root": r[2],
            "parent_session_id": r[3] or "",
            "owner": r[4] or "",
            "role": r[5] or "owner",
            "created_at": r[6],
            "updated_at": r[7],
            "message_count": r[8],
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
            SELECT DISTINCT s.id, s.title, s.workspace_root, s.parent_session_id, s.owner, s.role, s.created_at, s.updated_at, s.message_count
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
            "parent_session_id": r[3] or "",
            "owner": r[4] or "",
            "role": r[5] or "owner",
            "created_at": r[6],
            "updated_at": r[7],
            "message_count": r[8],
        }
        for r in rows
    ]


def list_child_sessions(
    parent_session_id: str, limit: int = 100
) -> list[dict[str, Any]]:
    """List sessions whose parent is *parent_session_id*."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT id, title, workspace_root, parent_session_id, owner, role, created_at, updated_at, message_count FROM sessions WHERE parent_session_id = ? ORDER BY updated_at DESC LIMIT ?",
            (parent_session_id, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "workspace_root": r[2],
            "parent_session_id": r[3] or "",
            "owner": r[4] or "",
            "role": r[5] or "owner",
            "created_at": r[6],
            "updated_at": r[7],
            "message_count": r[8],
        }
        for r in rows
    ]


def list_session_ancestors(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Walk parent links from *session_id* upwards, nearest parent first."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = get_session_info(session_id)
    while current and current.get("parent_session_id") and len(out) < limit:
        parent_id = str(current.get("parent_session_id") or "").strip()
        if not parent_id or parent_id in seen:
            break
        seen.add(parent_id)
        parent = get_session_info(parent_id)
        if parent is None:
            break
        out.append(parent)
        current = parent
    return out


def get_session_timeline(session_id: str) -> dict[str, Any] | None:
    """Return a combined timeline view for a session."""
    info = get_session_info(session_id)
    if info is None:
        return None
    from .checkpoints import list_checkpoints

    return {
        "session": info,
        "ancestors": list_session_ancestors(session_id),
        "children": list_child_sessions(session_id),
        "todos": load_session_todos(session_id) or [],
        "checkpoints": list_checkpoints(session_id, limit=20),
        "messages": load_session(session_id) or [],
    }


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
    new_id = create_session(
        title=new_title,
        workspace_root=workspace_root,
        parent_session_id=session_id,
        owner=info.get("owner") if info else None,
        role=info.get("role", "owner") if info else "owner",
    )
    save_session(new_id, new_title, messages, workspace_root=workspace_root)
    todos = load_session_todos(session_id)
    if todos:
        save_session_todos(new_id, todos)
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
    new_id = create_session(
        title=new_title,
        workspace_root=workspace_root,
        parent_session_id=session_id,
        owner=info.get("owner") if info else None,
        role=info.get("role", "owner") if info else "owner",
    )
    save_session(new_id, new_title, forked_messages, workspace_root=workspace_root)
    todos = load_session_todos(session_id)
    if todos:
        save_session_todos(new_id, todos)
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


def delete_session(session_id: str) -> bool:
    """Delete a session, its messages, and any stored todos."""
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        conn.execute("BEGIN")
        try:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                conn.execute("ROLLBACK")
                return False
            conn.execute(
                "DELETE FROM session_todos WHERE session_id = ?", (session_id,)
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.execute("COMMIT")
            try:
                from .control_plane import publish_event

                publish_event("session.deleted", {"session_id": session_id})
            except Exception:
                pass
            return True
        except BaseException:
            conn.execute("ROLLBACK")
            raise


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
        "todos": load_session_todos(session_id) or [],
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
    todos = data.get("todos")
    if todos is not None and not isinstance(todos, list):
        raise ValueError("Invalid 'todos' list in export data")

    session_meta = data.get("session") or {}
    import_title = (
        (title or "").strip()
        or (session_meta.get("title") or "").strip()
        or "Imported session"
    )
    workspace_root = session_meta.get("workspace_root") or ""

    new_id = create_session(title=import_title, workspace_root=workspace_root)
    save_session(new_id, import_title, messages, workspace_root=workspace_root)
    if todos:
        save_session_todos(new_id, todos)

    logger.info(
        "Imported session %s with %d messages (title: %s)",
        new_id,
        len(messages),
        import_title,
    )
    return new_id
