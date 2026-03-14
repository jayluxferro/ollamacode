from __future__ import annotations

from collections import deque

from ollamacode.control_plane import (
    list_recent_events,
    publish_event,
    subscribe,
    unsubscribe,
)


async def _next(queue):
    return await queue.get()


def test_control_plane_publish_subscribe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ollamacode.control_plane._EVENTS_PATH", tmp_path / "events.jsonl"
    )
    monkeypatch.setattr("ollamacode.control_plane._loaded_from_disk", False)
    monkeypatch.setattr("ollamacode.control_plane._recent_events", deque(maxlen=200))
    queue = subscribe()
    try:
        publish_event("session.created", {"session_id": "abc"})
        item = queue.get_nowait()
        assert item["type"] == "session.created"
        assert item["payload"]["session_id"] == "abc"
    finally:
        unsubscribe(queue)

def test_control_plane_recent_events(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ollamacode.control_plane._EVENTS_PATH", tmp_path / "events.jsonl"
    )
    monkeypatch.setattr("ollamacode.control_plane._loaded_from_disk", False)
    monkeypatch.setattr("ollamacode.control_plane._recent_events", deque(maxlen=200))
    publish_event("workspace.created", {"workspace_id": "w1"})
    recent = list_recent_events(limit=5)
    assert any(item["type"] == "workspace.created" for item in recent)


def test_control_plane_persists_events_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ollamacode.control_plane._EVENTS_PATH", tmp_path / "events.jsonl"
    )
    monkeypatch.setattr("ollamacode.control_plane._loaded_from_disk", False)
    monkeypatch.setattr("ollamacode.control_plane._recent_events", deque(maxlen=200))
    publish_event("workspace.updated", {"workspace_id": "w1"})
    # Simulate a fresh process view.
    monkeypatch.setattr("ollamacode.control_plane._loaded_from_disk", False)
    monkeypatch.setattr("ollamacode.control_plane._recent_events", deque(maxlen=200))
    recent = list_recent_events(limit=5)
    assert any(item["type"] == "workspace.updated" for item in recent)
