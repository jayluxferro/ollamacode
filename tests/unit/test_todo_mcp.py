"""Unit tests for the session todo MCP server."""

from __future__ import annotations

import json

import pytest

from ollamacode.sessions import create_session
from ollamacode.servers import todo_mcp


def test_todoread_requires_session_id(monkeypatch):
    """Todo tools should fail clearly without a session context."""
    monkeypatch.delenv("OLLAMACODE_SESSION_ID", raising=False)
    with pytest.raises(ValueError):
        todo_mcp.todoread()


def test_todowrite_round_trip(tmp_path, monkeypatch):
    """todowrite persists todos for the active session and todoread returns them."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    session_id = create_session("Todo MCP", workspace_root=str(tmp_path))
    monkeypatch.setenv("OLLAMACODE_SESSION_ID", session_id)

    written = todo_mcp.todowrite(
        [
            {"content": "Investigate agent todo plumbing", "status": "in_progress"},
            {"content": "Ship tests", "done": True, "priority": "high"},
        ]
    )
    read_back = todo_mcp.todoread()

    assert json.loads(written) == json.loads(read_back)
    assert json.loads(read_back) == [
        {
            "content": "Investigate agent todo plumbing",
            "status": "in_progress",
            "priority": "medium",
        },
        {
            "content": "Ship tests",
            "status": "completed",
            "priority": "high",
        },
    ]
