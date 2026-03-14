"""Unit tests for the CLI (arg parsing, env, run-one-query path with mocked agent)."""

from unittest.mock import AsyncMock, patch

import pytest

from ollamacode.cli import (
    _ensure_interactive_session,
    _parse_args,
    _prepare_mcp_servers,
    _resolve_mcp_servers,
    _run,
)


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
    with patch(
        "sys.argv",
        ["ollamacode", "--mcp-args", "python", "server.py", "--", "what is 2+2"],
    ):
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


def test_parse_args_python():
    """_parse_args accepts --python for MCP interpreter override."""
    with patch("sys.argv", ["ollamacode", "--python", "/usr/bin/python3"]):
        args = _parse_args()
    assert getattr(args, "python", None) == "/usr/bin/python3"
    with patch("sys.argv", ["ollamacode"]):
        args = _parse_args()
    assert getattr(args, "python", None) is None or getattr(args, "python", None) == ""


def test_parse_args_verbose():
    """_parse_args accepts --verbose."""
    with patch("sys.argv", ["ollamacode", "--verbose"]):
        args = _parse_args()
    assert getattr(args, "verbose", False) is True


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
        patch(
            "ollamacode.cli.run_agent_loop_no_mcp",
            new_callable=AsyncMock,
            return_value="Hi back",
        ),
    ):
        await _run("test-model", [], "", "hello", False, False, 0, None)
    out = capsys.readouterr().out
    assert "Hi back" in out


@pytest.mark.asyncio
async def test_run_respects_system_extra(monkeypatch):
    """_run appends system_extra to system prompt when set."""
    with patch("ollamacode.cli.run_agent_loop_no_mcp", new_callable=AsyncMock) as m:
        m.return_value = "ok"
        await _run(
            "test-model", [], "Extra instruction.", "hello", False, False, 0, None
        )
    m.assert_awaited_once()
    call_kw = m.call_args[1]
    assert "system_prompt" in call_kw
    assert "Extra instruction." in call_kw["system_prompt"]


@pytest.mark.asyncio
async def test_run_uses_mcp_args_from_env(monkeypatch):
    """_run prepares stdio MCP servers with the interactive auxiliaries attached."""
    mcp_servers = [
        {"type": "stdio", "command": "python", "args": ["examples/demo_server.py"]}
    ]
    with (
        patch("ollamacode.cli.connect_mcp_servers") as connect,
        patch(
            "ollamacode.cli.run_agent_loop", new_callable=AsyncMock, return_value="5"
        ),
    ):
        connect.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        connect.return_value.__aexit__ = AsyncMock(return_value=None)
        await _run("test-model", mcp_servers, "", "What is 2+3?", False, False, 0, None)
    connect.assert_called_once()
    call_args = connect.call_args[0]
    assert call_args[0][0]["command"] == "python"
    assert call_args[0][0]["args"] == ["examples/demo_server.py"]
    assert any(
        entry["args"] == ["-m", "ollamacode.servers.question_mcp"]
        for entry in call_args[0]
    )


def test_resolve_mcp_servers_returns_using_builtin(tmp_path):
    """_resolve_mcp_servers returns (servers, use_mcp, using_builtin)."""
    # No config file (missing path), no env -> using_builtin True
    servers, use_mcp, using_builtin = _resolve_mcp_servers(
        str(tmp_path / "missing.yaml"), "python", []
    )
    assert use_mcp is True
    assert using_builtin is True
    from ollamacode.config import DEFAULT_MCP_SERVERS

    assert len(servers) == len(DEFAULT_MCP_SERVERS)
    # With config file -> built-in servers prepended by default (include_builtin_servers True)
    cfg = tmp_path / "ollamacode.yaml"
    cfg.write_text("mcp_servers:\n  - type: stdio\n    command: npx\n    args: [mcp]\n")
    servers2, use_mcp2, using_builtin2 = _resolve_mcp_servers(str(cfg), "python", [])
    assert use_mcp2 is True
    assert using_builtin2 is False
    # Default: built-in + custom (1); custom server is last
    assert len(servers2) == len(DEFAULT_MCP_SERVERS) + 1
    assert servers2[-1]["command"] == "npx"


def test_prepare_mcp_servers_injects_session_env_and_todo_server():
    """Session-aware MCP prep should inject env and append the todo server."""
    servers = _prepare_mcp_servers(
        [{"type": "stdio", "command": "python", "args": ["-m", "ollamacode.servers.fs_mcp"]}],
        "/tmp/workspace",
        session_id="session-123",
        python_executable="/usr/bin/python3",
        subagents=[{"name": "reviewer", "tools": ["read_file"]}],
    )
    assert len(servers) == 4
    assert servers[0]["env"]["OLLAMACODE_FS_ROOT"] == "/tmp/workspace"
    assert servers[0]["env"]["OLLAMACODE_SESSION_ID"] == "session-123"
    assert servers[1]["command"] == "/usr/bin/python3"
    assert servers[1]["args"] == ["-m", "ollamacode.servers.question_mcp"]
    assert servers[2]["args"] == ["-m", "ollamacode.servers.task_mcp"]
    assert servers[3]["args"] == ["-m", "ollamacode.servers.todo_mcp"]
    assert servers[3]["env"]["OLLAMACODE_SESSION_ID"] == "session-123"


def test_prepare_mcp_servers_without_session_does_not_append_todo_server():
    """Stateless prep should expose question, but not todo or task by default."""
    servers = _prepare_mcp_servers(
        [{"type": "stdio", "command": "python", "args": ["demo.py"]}],
        "/tmp/workspace",
    )
    assert len(servers) == 2
    assert "OLLAMACODE_SESSION_ID" not in servers[0]["env"]
    assert servers[1]["args"] == ["-m", "ollamacode.servers.question_mcp"]


def test_ensure_interactive_session_bootstraps_when_query_missing(tmp_path, monkeypatch):
    """Interactive mode should auto-create a session so session-scoped tools can work."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    session_id, session_history = _ensure_interactive_session(
        None,
        None,
        "Interactive",
        str(tmp_path),
        None,
    )
    assert session_id
    assert session_history == []
