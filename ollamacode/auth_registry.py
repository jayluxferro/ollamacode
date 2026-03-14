"""Simple persisted principal/token registry for the control plane."""

from __future__ import annotations

import json
import secrets
import uuid
from pathlib import Path
from typing import Any

_AUTH_PATH = Path.home() / ".ollamacode" / "principals.json"


def _path() -> Path:
    _AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _AUTH_PATH


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


def list_principals() -> list[dict[str, Any]]:
    return sorted(_load(), key=lambda row: str(row.get("name") or row.get("id") or ""))


def get_principal(principal_id: str) -> dict[str, Any] | None:
    for row in _load():
        if row.get("id") == principal_id:
            return row
    return None


def create_principal(
    *,
    name: str,
    role: str = "admin",
    api_key: str = "",
    workspace_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows = _load()
    entry = {
        "id": str(uuid.uuid4()),
        "name": name.strip() or "Principal",
        "role": role.strip() or "admin",
        "api_key": api_key.strip() or secrets.token_urlsafe(24),
        "workspace_ids": list(workspace_ids or []),
    }
    rows.append(entry)
    _save(rows)
    return entry


def update_principal(
    principal_id: str,
    *,
    name: str | None = None,
    role: str | None = None,
    api_key: str | None = None,
    workspace_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    rows = _load()
    for row in rows:
        if row.get("id") != principal_id:
            continue
        if name is not None:
            row["name"] = name.strip() or row.get("name") or "Principal"
        if role is not None:
            row["role"] = role.strip() or row.get("role") or "admin"
        if api_key is not None:
            row["api_key"] = (
                api_key.strip() or row.get("api_key") or secrets.token_urlsafe(24)
            )
        if workspace_ids is not None:
            row["workspace_ids"] = list(workspace_ids)
        _save(rows)
        return row
    return None


def delete_principal(principal_id: str) -> bool:
    rows = _load()
    filtered = [row for row in rows if row.get("id") != principal_id]
    if len(filtered) == len(rows):
        return False
    _save(filtered)
    return True


def find_principal_by_token(token: str) -> dict[str, Any] | None:
    token = (token or "").strip()
    if not token:
        return None
    for row in _load():
        if row.get("api_key") == token:
            return row
    return None
