"""Unit tests for the CLI (arg parsing, env, run-one-query path with mocked agent)."""

from unittest.mock import AsyncMock, patch

import pytest

from ollamacode.cli import _parse_args, _run


def test_parse_args_defaults():
    """_parse_args uses default model and no MCP args when not provided."""
    with patch("sys.argv", ["ollamacode"]):
        args = _parse_args()
    assert args.model == "gpt-oss:20b"  # or env value
    assert args.mcp_command == "python"
    assert args.mcp_args == []
    assert args.query is None


def test_parse_args_query_and_model():
    """_parse_args accepts positional query and -m model."""
    with patch("sys.argv", ["ollamacode", "-m", "llama3.2", "hello"]):
        args = _parse_args()
    assert args.model == "llama3.2"
    assert args.query == "hello"


def test_parse_args_mcp_args():
    """_parse_args accepts --mcp-args; use -- to separate from query."""
    with patch("sys.argv", ["ollamacode", "--mcp-args", "python", "server.py", "--", "what is 2+2"]):
        args = _parse_args()
    assert args.mcp_args == ["python", "server.py"]
    assert args.query == "what is 2+2"


def test_parse_args_stream():
    """_parse_args accepts --stream / -s."""
    with patch("sys.argv", ["ollamacode", "--stream", "hello"]):
        args = _parse_args()
    assert args.stream is True
    assert args.query == "hello"
    with patch("sys.argv", ["ollamacode", "-s", "hi"]):
        args = _parse_args()
    assert args.stream is True


def test_parse_args_tui():
    """_parse_args accepts --tui."""
    with patch("sys.argv", ["ollamacode", "--tui"]):
        args = _parse_args()
    assert args.tui is True
    assert args.query is None
    with patch("sys.argv", ["ollamacode", "--tui", "single query"]):
        args = _parse_args()
    assert args.tui is True
    assert args.query == "single query"


def test_parse_args_env_model(monkeypatch):
    """_parse_args uses OLLAMACODE_MODEL when set."""
    monkeypatch.setenv("OLLAMACODE_MODEL", "custom-model")
    with patch("sys.argv", ["ollamacode"]):
        args = _parse_args()
    assert args.model == "custom-model"
    monkeypatch.delenv("OLLAMACODE_MODEL", raising=False)


@pytest.mark.asyncio
async def test_run_no_mcp_single_query(capsys):
    """_run with query and no MCP prints agent output."""
    with (
        patch("ollamacode.cli.run_agent_loop_no_mcp", new_callable=AsyncMock, return_value="Hi back"),
    ):
        await _run("test-model", [], "", "hello", False, False, 0, None)
    out = capsys.readouterr().out
    assert "Hi back" in out


@pytest.mark.asyncio
async def test_run_respects_system_extra(monkeypatch):
    """_run appends system_extra to system prompt when set."""
    with patch("ollamacode.cli.run_agent_loop_no_mcp", new_callable=AsyncMock) as m:
        m.return_value = "ok"
        await _run("test-model", [], "Extra instruction.", "hello", False, False, 0, None)
    m.assert_awaited_once()
    call_kw = m.call_args[1]
    assert "system_prompt" in call_kw
    assert "Extra instruction." in call_kw["system_prompt"]


@pytest.mark.asyncio
async def test_run_uses_mcp_args_from_env(monkeypatch):
    """_run with single stdio MCP server config uses connect_mcp_stdio."""
    mcp_servers = [{"type": "stdio", "command": "python", "args": ["examples/demo_server.py"]}]
    with (
        patch("ollamacode.cli.connect_mcp_stdio") as connect,
        patch("ollamacode.cli.run_agent_loop", new_callable=AsyncMock, return_value="5"),
    ):
        connect.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        connect.return_value.__aexit__ = AsyncMock(return_value=None)
        await _run("test-model", mcp_servers, "", "What is 2+3?", False, False, 0, None)
    connect.assert_called_once()
    call_args = connect.call_args[0]
    assert call_args[0] == "python"
    assert call_args[1] == ["examples/demo_server.py"]
