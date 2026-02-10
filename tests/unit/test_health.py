"""Unit tests for health check (check_ollama)."""

from unittest.mock import patch

import ollama

from ollamacode.health import check_ollama


def test_check_ollama_success():
    """check_ollama returns (True, message) when ollama.list() succeeds."""
    with patch.object(ollama, "list", return_value={"models": []}):
        ok, msg = check_ollama()
    assert ok is True
    assert "reachable" in msg.lower()


def test_check_ollama_connection_refused():
    """check_ollama returns (False, message) when connection refused."""
    with patch.object(ollama, "list", side_effect=Exception("connection refused")):
        ok, msg = check_ollama()
    assert ok is False
    assert "not running" in msg.lower() or "ollama serve" in msg
