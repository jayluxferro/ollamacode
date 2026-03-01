"""Unit tests for TUI streaming text sanitization."""

from ollamacode.tui import _sanitize_stream_text


def test_sanitize_stream_text_removes_ansi_and_cr() -> None:
    raw = "\x1b[31mHello\x1b[0m\rWorld\r\nDone\x00"
    out = _sanitize_stream_text(raw)
    assert "\x1b" not in out
    assert "\r" not in out
    assert "\x00" not in out
    assert "Hello" in out
    assert "World" in out
    assert "Done" in out
