"""Lightweight in-process control-plane event stream."""

from __future__ import annotations

import asyncio
from collections import deque
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
_recent_events: deque[dict[str, Any]] = deque(maxlen=200)
_EVENTS_PATH = Path.home() / ".ollamacode" / "control_plane_events.jsonl"
_loaded_from_disk = False


def _path() -> Path:
    _EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _EVENTS_PATH


def _ensure_loaded() -> None:
    global _loaded_from_disk
    if _loaded_from_disk:
        return
    _loaded_from_disk = True
    path = _path()
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines[-200:]:
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict):
            _recent_events.append(event)


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    _ensure_loaded()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue[dict[str, Any]]) -> None:
    _subscribers.discard(queue)


def list_recent_events(limit: int = 50) -> list[dict[str, Any]]:
    _ensure_loaded()
    limit = max(1, min(limit, 200))
    return list(_recent_events)[-limit:]


def publish_event(event_type: str, payload: dict[str, Any]) -> None:
    _ensure_loaded()
    event = {
        "type": event_type,
        "payload": payload,
        "time": {"created": datetime.now(timezone.utc).isoformat()},
    }
    _recent_events.append(event)
    try:
        with open(_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass
    for queue in list(_subscribers):
        try:
            queue.put_nowait(event)
        except Exception:
            _subscribers.discard(queue)
