"""Unit tests for centralised Ollama client (chat + generate fallback)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ollamacode.ollama_client import (
    chat_async,
    chat_sync,
    chat_stream_sync,
    is_ollama_template_error,
    wrap_ollama_template_error,
)


def test_is_ollama_template_error():
    """Template error detection matches known Ollama message (must include 'template')."""
    assert (
        is_ollama_template_error(ValueError("template slice index out of range"))
        is True
    )
    assert is_ollama_template_error(ValueError("template reflect: slice index")) is True
    assert is_ollama_template_error(ValueError("template error index $prop")) is True
    assert is_ollama_template_error(ValueError("other error")) is False


def test_wrap_ollama_template_error():
    """Template errors get hint; others pass through."""
    e = ValueError("template slice index out of range")
    wrapped = wrap_ollama_template_error(e)
    assert isinstance(wrapped, RuntimeError)
    assert "Try --no-mcp" in str(wrapped)
    other = ValueError("network error")
    assert wrap_ollama_template_error(other) is other


def test_chat_sync_returns_chat_shape():
    """chat_sync returns message content in chat shape when ollama.chat succeeds."""
    fake = {"message": {"content": "Hi there"}}
    with patch("ollamacode.ollama_client.ollama") as mock_ollama:
        mock_ollama.chat.return_value = fake
        out = chat_sync("test-model", [{"role": "user", "content": "Hi"}], [])
    assert out == fake
    mock_ollama.chat.assert_called_once()


def test_chat_sync_fallback_on_template_error():
    """chat_sync falls back to generate API when chat raises template error and tools empty."""
    with patch("ollamacode.ollama_client.ollama") as mock_ollama:
        mock_ollama.chat.side_effect = ValueError("template slice index out of range")
        mock_ollama.generate.return_value = {"response": "Generated reply"}
        out = chat_sync("test-model", [{"role": "user", "content": "Hi"}], [])
    assert out == {"message": {"content": "Generated reply"}}
    mock_ollama.generate.assert_called_once()


def test_chat_sync_no_fallback_when_tools():
    """chat_sync does not fall back when tools are non-empty; re-raises wrapped."""
    with patch("ollamacode.ollama_client.ollama") as mock_ollama:
        mock_ollama.chat.side_effect = ValueError("template slice index")
        with pytest.raises(RuntimeError) as exc_info:
            chat_sync(
                "test-model",
                [{"role": "user", "content": "Hi"}],
                [{"function": {"name": "x"}}],
            )
    assert "Try --no-mcp" in str(exc_info.value)
    mock_ollama.generate.assert_not_called()


def test_chat_stream_sync_yields_content():
    """chat_stream_sync yields (content,) per chunk."""
    with patch("ollamacode.ollama_client.ollama") as mock_ollama:
        mock_ollama.chat.return_value = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
        ]
        chunks = list(
            chat_stream_sync("test-model", [{"role": "user", "content": "Hi"}])
        )
    assert chunks == [("Hello",), (" world",)]


@pytest.mark.asyncio
async def test_chat_async_returns_chat_shape():
    """chat_async uses AsyncClient and returns dict with message."""
    fake = {"message": {"content": "Hi there"}}
    with patch("ollamacode.ollama_client.ollama") as mock_ollama:
        mock_client = AsyncMock()
        response_obj = MagicMock()
        response_obj.model_dump.return_value = fake
        mock_client.chat = AsyncMock(return_value=response_obj)
        mock_client.close = AsyncMock()
        mock_client.generate = AsyncMock()
        mock_ollama.AsyncClient.return_value = mock_client
        out = await chat_async("test-model", [{"role": "user", "content": "Hi"}], [])
    assert out == fake
    mock_client.chat.assert_awaited_once()
    mock_client.close.assert_awaited_once()
