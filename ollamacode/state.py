"""
Persistent state: recent files, preferences, etc. Stored in ~/.ollamacode/state.json.

Used so the assistant can remember context across sessions. MCP tools (state_mcp) expose get/update/clear.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_STATE_PATH = Path(os.path.expanduser("~")) / ".ollamacode" / "state.json"
_MAX_RECENT_FILES = 50


def _load_raw() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_state() -> dict:
    """Return full state dict (recent_files, preferences, etc.)."""
    return _load_raw()


def update_state(
    recent_files: list[str] | None = None,
    preferences: dict | None = None,
    **kwargs: str | list[str] | dict,
) -> str:
    """
    Update state. recent_files: replace with this list (or append if key is append_recent_files).
    preferences: merge into state["preferences"]. Other kwargs merged at top level.
    Returns a short status message.
    """
    data = _load_raw()
    if recent_files is not None:
        data["recent_files"] = list(recent_files)[-_MAX_RECENT_FILES:]
    if preferences is not None:
        prefs = data.get("preferences") or {}
        prefs.update(preferences)
        data["preferences"] = prefs
    for k, v in kwargs.items():
        if k in ("recent_files", "preferences"):
            continue
        data[k] = v
    _save(data)
    return f"Updated state ({len(data)} keys) in {_STATE_PATH}"


def append_recent_file(path: str) -> str:
    """Append a file path to recent_files (dedupe, keep last MAX)."""
    data = _load_raw()
    recent = data.get("recent_files") or []
    if path in recent:
        recent.remove(path)
    recent.append(path)
    data["recent_files"] = recent[-_MAX_RECENT_FILES:]
    _save(data)
    return "ok"


def clear_state() -> str:
    """Clear all state. Returns status message."""
    _save({})
    return f"Cleared state at {_STATE_PATH}"


def format_recent_context(state: dict, max_files: int = 10) -> str:
    """
    Format recent_files from state for injection into the system prompt.
    Returns a short block (e.g. "Recent files: path1, path2, ...") or "" if empty.
    """
    recent = state.get("recent_files") or []
    if not recent or max_files <= 0:
        return ""
    take = recent[-max_files:]
    return "Recent files: " + ", ".join(take)
