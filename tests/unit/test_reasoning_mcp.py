"""Unit tests for the reasoning MCP server (reasoning_mcp.py).

Covers: think(), get_think_buffer(), clear_think_buffer(),
per-session isolation, buffer size limits.
"""

import pytest

from ollamacode.servers import reasoning_mcp


@pytest.fixture(autouse=True)
def _clean_buffers():
    """Clear all think buffers before and after each test."""
    reasoning_mcp._THINK_BUFFERS.clear()
    yield
    reasoning_mcp._THINK_BUFFERS.clear()


class TestThink:
    def test_think_returns_ok(self, monkeypatch):
        """think() returns 'OK' for non-empty reasoning."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "s1")
        assert reasoning_mcp.think("Step 1: analyze the problem") == "OK"

    def test_think_appends_to_buffer(self, monkeypatch):
        """think() appends entries to the session buffer."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "s1")
        reasoning_mcp.think("thought A")
        reasoning_mcp.think("thought B")
        buf = reasoning_mcp.get_think_buffer("s1")
        assert buf == ["thought A", "thought B"]

    def test_think_ignores_empty(self, monkeypatch):
        """think() ignores empty or whitespace-only reasoning."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "s1")
        reasoning_mcp.think("")
        reasoning_mcp.think("   ")
        buf = reasoning_mcp.get_think_buffer("s1")
        assert buf == []


class TestBufferOverflow:
    def test_buffer_capped_at_max(self, monkeypatch):
        """Buffer is capped at _THINK_BUFFER_MAX (10) entries, FIFO eviction."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "overflow")
        for i in range(15):
            reasoning_mcp.think(f"step {i}")
        buf = reasoning_mcp.get_think_buffer("overflow")
        assert len(buf) == reasoning_mcp._THINK_BUFFER_MAX
        # Oldest entries (0-4) should be evicted, newest (5-14) retained
        assert buf[0] == "step 5"
        assert buf[-1] == "step 14"


class TestSessionIsolation:
    def test_separate_sessions(self, monkeypatch):
        """Different session IDs maintain independent buffers."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "alpha")
        reasoning_mcp.think("alpha thought")

        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "beta")
        reasoning_mcp.think("beta thought")

        assert reasoning_mcp.get_think_buffer("alpha") == ["alpha thought"]
        assert reasoning_mcp.get_think_buffer("beta") == ["beta thought"]

    def test_default_session_id(self, monkeypatch):
        """When OLLAMACODE_SESSION_ID is not set, the default '' key is used."""
        monkeypatch.delenv("OLLAMACODE_SESSION_ID", raising=False)
        reasoning_mcp.think("default thought")
        buf = reasoning_mcp.get_think_buffer("")
        assert buf == ["default thought"]


class TestClearBuffer:
    def test_clear_removes_entries(self, monkeypatch):
        """clear_think_buffer() empties the session buffer."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "s1")
        reasoning_mcp.think("thought")
        reasoning_mcp.clear_think_buffer("s1")
        assert reasoning_mcp.get_think_buffer("s1") == []

    def test_clear_nonexistent_session_is_noop(self):
        """clear_think_buffer for a nonexistent session does not raise."""
        reasoning_mcp.clear_think_buffer("nonexistent")

    def test_clear_does_not_affect_other_sessions(self, monkeypatch):
        """Clearing one session does not affect another."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "keep")
        reasoning_mcp.think("keep me")
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "clear")
        reasoning_mcp.think("delete me")

        reasoning_mcp.clear_think_buffer("clear")
        assert reasoning_mcp.get_think_buffer("keep") == ["keep me"]
        assert reasoning_mcp.get_think_buffer("clear") == []


class TestGetThinkBuffer:
    def test_returns_copy(self, monkeypatch):
        """get_think_buffer returns a copy, not the internal list."""
        monkeypatch.setenv("OLLAMACODE_SESSION_ID", "s1")
        reasoning_mcp.think("thought")
        buf = reasoning_mcp.get_think_buffer("s1")
        buf.append("extra")
        assert reasoning_mcp.get_think_buffer("s1") == ["thought"]

    def test_empty_for_unknown_session(self):
        """get_think_buffer returns [] for unknown session IDs."""
        assert reasoning_mcp.get_think_buffer("no_such_session") == []
