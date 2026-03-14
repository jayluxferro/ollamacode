"""Workspace registry for local and remote OllamaCode workspaces."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

_WORKSPACES_PATH = Path.home() / ".ollamacode" / "workspaces.json"


def _path() -> Path:
    _WORKSPACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _WORKSPACES_PATH


def _load() -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _save(rows: list[dict[str, Any]]) -> None:
    _path().write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def list_workspaces() -> list[dict[str, Any]]:
    return sorted(_load(), key=lambda row: str(row.get("name") or row.get("id") or ""))


def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    for row in _load():
        if row.get("id") == workspace_id:
            return row
    return None


def create_workspace(
    *,
    name: str,
    kind: str = "local",
    workspace_root: str = "",
    base_url: str = "",
    api_key: str = "",
    owner: str = "",
    role: str = "owner",
) -> dict[str, Any]:
    rows = _load()
    entry = {
        "id": str(uuid.uuid4()),
        "name": name.strip() or "Workspace",
        "type": kind.strip().lower() or "local",
        "workspace_root": workspace_root.strip(),
        "base_url": base_url.strip(),
        "api_key": api_key.strip(),
        "owner": owner.strip(),
        "role": role.strip() or "owner",
        "last_status": "unknown",
        "last_error": "",
    }
    rows.append(entry)
    _save(rows)
    try:
        from .control_plane import publish_event

        publish_event("workspace.created", {"workspace": entry})
    except Exception:
        pass
    return entry


def update_workspace(
    workspace_id: str,
    *,
    name: str | None = None,
    kind: str | None = None,
    workspace_root: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    owner: str | None = None,
    role: str | None = None,
    last_status: str | None = None,
    last_error: str | None = None,
) -> dict[str, Any] | None:
    rows = _load()
    for row in rows:
        if row.get("id") != workspace_id:
            continue
        if name is not None:
            row["name"] = name.strip() or row.get("name") or "Workspace"
        if kind is not None:
            row["type"] = kind.strip().lower() or row.get("type") or "local"
        if workspace_root is not None:
            row["workspace_root"] = workspace_root.strip()
        if base_url is not None:
            row["base_url"] = base_url.strip()
        if api_key is not None:
            row["api_key"] = api_key.strip()
        if owner is not None:
            row["owner"] = owner.strip()
        if role is not None:
            row["role"] = role.strip() or row.get("role") or "owner"
        if last_status is not None:
            row["last_status"] = last_status.strip()
        if last_error is not None:
            row["last_error"] = last_error.strip()
        _save(rows)
        try:
            from .control_plane import publish_event

            publish_event("workspace.updated", {"workspace": row})
        except Exception:
            pass
        return row
    return None


def delete_workspace(workspace_id: str) -> bool:
    rows = _load()
    filtered = [row for row in rows if row.get("id") != workspace_id]
    if len(filtered) == len(rows):
        return False
    _save(filtered)
    try:
        from .control_plane import publish_event

        publish_event("workspace.deleted", {"workspace_id": workspace_id})
    except Exception:
        pass
    return True
