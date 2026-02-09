"""Unit tests for editor protocol (normalize_chat_body)."""

from ollamacode.protocol import normalize_chat_body


def test_normalize_message_only():
    """Only message is required."""
    msg, file_path, lines = normalize_chat_body({"message": "hello"})
    assert msg == "hello"
    assert file_path is None
    assert lines is None


def test_normalize_file_and_lines():
    """file and lines are passed through."""
    msg, file_path, lines = normalize_chat_body(
        {
            "message": "explain this",
            "file": "src/foo.py",
            "lines": "10-20",
        }
    )
    assert msg == "explain this"
    assert file_path == "src/foo.py"
    assert lines == "10-20"


def test_normalize_selection_overrides_file_lines():
    """selection overrides file/lines when both present."""
    msg, file_path, lines = normalize_chat_body(
        {
            "message": "fix this",
            "file": "other.py",
            "lines": "1-2",
            "selection": {"file": "src/bar.py", "startLine": 5, "endLine": 10},
        }
    )
    assert msg == "fix this"
    assert file_path == "src/bar.py"
    assert lines == "5-10"


def test_normalize_selection_only():
    """selection alone sets file and lines."""
    msg, file_path, lines = normalize_chat_body(
        {
            "message": "what does this do?",
            "selection": {"file": "a/b.py", "startLine": 1, "endLine": 1},
        }
    )
    assert msg == "what does this do?"
    assert file_path == "a/b.py"
    assert lines == "1-1"


def test_normalize_selection_invalid_ignored():
    """Non-dict or empty selection is ignored."""
    msg, file_path, lines = normalize_chat_body(
        {
            "message": "hi",
            "selection": [],
        }
    )
    assert msg == "hi"
    assert file_path is None
    assert lines is None

    msg2, fp2, ln2 = normalize_chat_body(
        {
            "message": "hi",
            "selection": {"file": "x", "startLine": "not int"},
        }
    )
    assert msg2 == "hi"
    assert fp2 == "x"
    assert ln2 is None
