"""Unit tests for health check (check_ollama, check_toolchain_versions)."""

from unittest.mock import patch

import ollama

from ollamacode.health import check_ollama, check_toolchain_versions


def test_check_toolchain_versions_empty():
    """check_toolchain_versions returns [] when no checks."""
    assert check_toolchain_versions([]) == []


def test_check_toolchain_versions_ok(tmp_path):
    """check_toolchain_versions reports ok when output contains expect_contains."""
    checks = [
        {
            "name": "py",
            "command": "python -c \"print('pytest 7.4')\"",
            "expect_contains": "7",
        }
    ]
    results = check_toolchain_versions(checks, cwd=tmp_path)
    assert len(results) == 1
    assert results[0]["name"] == "py"
    assert results[0]["ok"] is True
    assert "7" in results[0]["actual"]


def test_check_toolchain_versions_mismatch(tmp_path):
    """check_toolchain_versions reports ok=False when expect_contains not in output."""
    checks = [
        {
            "name": "py",
            "command": "python -c \"print('hello')\"",
            "expect_contains": "7",
        }
    ]
    results = check_toolchain_versions(checks, cwd=tmp_path)
    assert len(results) == 1
    assert results[0]["ok"] is False


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
