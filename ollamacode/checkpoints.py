"""
Checkpointing: record file changes per turn and allow rewind.

Stored in ~/.ollamacode/checkpoints.db (SQLite).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DB_PATH = Path.home() / ".ollamacode" / "checkpoints.db"
_MAX_CONTENT_LEN = 1_000_000


def _db_path() -> Path:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            workspace_root TEXT NOT NULL,
            created_at REAL NOT NULL,
            prompt TEXT NOT NULL,
            message_index INTEGER NOT NULL DEFAULT 0,
            file_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS checkpoint_files (
            checkpoint_id TEXT NOT NULL,
            path TEXT NOT NULL,
            before_content TEXT,
            after_content TEXT,
            before_exists INTEGER NOT NULL,
            after_exists INTEGER NOT NULL,
            PRIMARY KEY (checkpoint_id, path),
            FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
        );
        CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id, created_at DESC);
        """
    )
    conn.commit()
    try:
        _DB_PATH.chmod(0o600)
    except OSError:
        pass


def _safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@dataclass
class FileSnapshot:
    before_content: str | None
    after_content: str | None
    before_exists: bool
    after_exists: bool


class CheckpointRecorder:
    def __init__(
        self,
        session_id: str,
        workspace_root: str,
        prompt: str,
        message_index: int,
    ) -> None:
        self.session_id = session_id
        self.workspace_root = workspace_root
        self.prompt = prompt
        self.message_index = message_index
        self._before: dict[str, FileSnapshot] = {}

    def _resolve(self, path: str) -> Path | None:
        try:
            root = Path(self.workspace_root).resolve()
            p = (root / path.lstrip("/")).resolve()
            if not p.is_relative_to(root):
                return None
            return p
        except Exception:
            return None

    def record_pre(self, path: str) -> None:
        if not path or path in self._before:
            return
        p = self._resolve(path)
        if p is None:
            return
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""
            self._before[path] = FileSnapshot(
                before_content=content[:_MAX_CONTENT_LEN],
                after_content=None,
                before_exists=True,
                after_exists=False,
            )
        else:
            self._before[path] = FileSnapshot(
                before_content=None,
                after_content=None,
                before_exists=False,
                after_exists=False,
            )

    def finalize(self) -> str | None:
        if not self._before:
            return None
        for path, snap in list(self._before.items()):
            p = self._resolve(path)
            if p is None:
                continue
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = ""
                snap.after_content = content[:_MAX_CONTENT_LEN]
                snap.after_exists = True
            else:
                snap.after_content = None
                snap.after_exists = False
        checkpoint_id = str(uuid.uuid4())
        with sqlite3.connect(_db_path()) as conn:
            _init_schema(conn)
            conn.execute(
                "INSERT INTO checkpoints (id, session_id, workspace_root, created_at, prompt, message_index, file_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    checkpoint_id,
                    self.session_id,
                    self.workspace_root,
                    time.time(),
                    (self.prompt or "")[:4000],
                    self.message_index,
                    len(self._before),
                ),
            )
            for path, snap in self._before.items():
                conn.execute(
                    "INSERT INTO checkpoint_files (checkpoint_id, path, before_content, after_content, before_exists, after_exists) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        checkpoint_id,
                        path,
                        snap.before_content,
                        snap.after_content,
                        1 if snap.before_exists else 0,
                        1 if snap.after_exists else 0,
                    ),
                )
            conn.commit()
        return checkpoint_id


def list_checkpoints(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT id, created_at, prompt, message_index, file_count FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "created_at": r[1],
            "prompt": r[2],
            "message_index": r[3],
            "file_count": r[4],
        }
        for r in rows
    ]


def get_checkpoint_files(checkpoint_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(_db_path()) as conn:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT path, before_content, after_content, before_exists, after_exists FROM checkpoint_files WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "path": r[0],
            "before_content": r[1],
            "after_content": r[2],
            "before_exists": bool(r[3]),
            "after_exists": bool(r[4]),
        }
        for r in rows
    ]


def restore_checkpoint(checkpoint_id: str, workspace_root: str) -> list[str]:
    """Restore file contents to the 'before' snapshot. Returns list of modified paths."""
    root = Path(workspace_root).resolve()
    modified: list[str] = []
    for item in get_checkpoint_files(checkpoint_id):
        path = item["path"]
        try:
            p = (root / path.lstrip("/")).resolve()
            if not p.is_relative_to(root):
                continue
        except Exception:
            continue
        before_exists = item.get("before_exists", False)
        before_content = item.get("before_content")
        if before_exists:
            _safe_write_text(p, before_content or "")
            modified.append(path)
        else:
            try:
                if p.exists():
                    p.unlink()
                    modified.append(path)
            except Exception:
                continue
    return modified
