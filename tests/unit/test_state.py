"""Unit tests for persistent state and format_recent_context."""

from unittest.mock import patch


from ollamacode.state import (
    format_recent_context,
    get_state,
)


def test_format_recent_context_empty():
    """format_recent_context returns '' for empty or no recent_files."""
    assert format_recent_context({}) == ""
    assert format_recent_context({"recent_files": []}) == ""
    assert format_recent_context({"recent_files": []}, max_files=0) == ""


def test_format_recent_context_respects_max_files():
    """format_recent_context returns at most max_files paths (last N)."""
    state = {"recent_files": ["a", "b", "c", "d", "e"]}
    assert format_recent_context(state, max_files=2) == "Recent files: d, e"
    assert format_recent_context(state, max_files=10) == "Recent files: a, b, c, d, e"


def test_format_recent_context_single():
    """format_recent_context with one file."""
    assert (
        format_recent_context({"recent_files": ["foo/bar.py"]})
        == "Recent files: foo/bar.py"
    )


def test_get_state_empty(tmp_path):
    """get_state returns {} when state file does not exist or is empty."""
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        assert get_state() == {}
    (tmp_path / "state.json").write_text("{}")
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        assert get_state() == {}
