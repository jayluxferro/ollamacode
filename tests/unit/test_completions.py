"""Unit tests for IDE completions (get_completion)."""

from unittest.mock import patch

from ollamacode.completions import get_completion


def test_get_completion_empty_model_returns_empty():
    """get_completion returns '' when model is empty or missing."""
    assert get_completion("def ", "", max_tokens=10) == ""
    assert get_completion("def ", "  ", max_tokens=10) == ""


def test_get_completion_success():
    """get_completion returns generated text when Ollama responds."""
    with patch("ollamacode.completions.ollama") as mock_ollama:
        mock_ollama.generate.return_value = type(
            "R", (), {"response": " foo(): pass"}
        )()
        out = get_completion("def ", "llama3.2", max_tokens=20)
    assert out == "foo(): pass"


def test_get_completion_truncates_multiline():
    """get_completion returns first line when response has newlines."""
    with patch("ollamacode.completions.ollama") as mock_ollama:
        mock_ollama.generate.return_value = type(
            "R", (), {"response": " line1\nline2"}
        )()
        out = get_completion("x", "m", max_tokens=10)
    assert out == "line1"
