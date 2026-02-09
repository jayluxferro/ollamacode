"""Unit tests for the agent loop (mocked Ollama and MCP)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from ollamacode.agent import (
    _parse_tool_args,
    _tool_call_one_line,
    run_agent_loop,
    run_agent_loop_no_mcp,
)


def test_parse_tool_args_tolerates_extra_brace():
    """_parse_tool_args fixes common LLM mistake: extra '}' at end of JSON."""
    raw = '{"cwd":".","message":"Update after test pass"}}'
    assert _parse_tool_args(raw) == {"cwd": ".", "message": "Update after test pass"}
    assert _parse_tool_args('{"a":1}}') == {"a": 1}
    assert _parse_tool_args("{}") == {}
    assert _parse_tool_args('{"x":1}') == {"x": 1}


def test_parse_tool_args_tolerates_extra_bracket_and_newlines():
    """_parse_tool_args fixes extra ']' before '}' and unescaped newlines in strings."""
    assert _parse_tool_args('{"key": "val"]}') == {"key": "val"}
    assert _parse_tool_args('{"a": "line1\n line2"}') == {"a": "line1  line2"}


def test_tool_call_one_line():
    """_tool_call_one_line produces one-line summaries for brief progress."""
    assert (
        _tool_call_one_line("read_file", {"path": "src/foo.py"})
        == "read_file(src/foo.py)"
    )
    assert (
        _tool_call_one_line("run_command", {"command": "pytest"})
        == "run_command(pytest)"
    )
    assert (
        _tool_call_one_line("run_command", {"command": "x" * 60})
        == "run_command(" + "x" * 50 + "...)"
    )
    assert _tool_call_one_line("list_dir", {"path": "/tmp"}) == "list_dir(/tmp)"
    assert _tool_call_one_line("unknown_tool", {}) == "unknown_tool"
    assert "bar" in _tool_call_one_line("other", {"bar": "baz"})


def _make_message(content: str, tool_calls: list | None = None) -> dict:
    msg = {"content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _make_tool_call(name: str, arguments: dict) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
async def test_run_agent_loop_no_mcp_returns_content():
    """run_agent_loop_no_mcp returns the model content when Ollama returns a message."""
    fake_response = MagicMock()
    fake_response.message = {"content": "Hello, world!"}

    with patch("ollamacode.agent._ollama_chat_sync", return_value=fake_response):
        out = await run_agent_loop_no_mcp(
            "test-model", "Hi", system_prompt="You are helpful."
        )
    assert out == "Hello, world!"


@pytest.mark.asyncio
async def test_run_agent_loop_no_mcp_no_message():
    """run_agent_loop_no_mcp returns 'No response from model.' when message is None."""
    with patch("ollamacode.agent._ollama_chat_sync", return_value={}):
        out = await run_agent_loop_no_mcp("test-model", "Hi")
    assert out == "No response from model."


@pytest.mark.asyncio
async def test_run_agent_loop_no_mcp_empty_content():
    """run_agent_loop_no_mcp returns stripped content (empty string if none)."""
    fake_response = MagicMock()
    fake_response.message = {"content": ""}

    with patch("ollamacode.agent._ollama_chat_sync", return_value=fake_response):
        out = await run_agent_loop_no_mcp("test-model", "Hi")
    assert out == ""


@pytest.mark.asyncio
async def test_run_agent_loop_single_turn_returns_text():
    """run_agent_loop returns model text when no tool_calls."""
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[]))
    session.call_tool = AsyncMock()

    fake_response = MagicMock()
    fake_response.message = _make_message("The answer is 42.")

    with patch("ollamacode.agent._ollama_chat_sync", return_value=fake_response):
        out = await run_agent_loop(session, "test-model", "What is the answer?")
    assert out == "The answer is 42."
    session.list_tools.assert_awaited_once()
    session.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_agent_loop_tool_call_then_text():
    """run_agent_loop runs tool then returns final model text."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        inputSchema={
            "type": "object",
            "required": ["a", "b"],
            "properties": {"a": {}, "b": {}},
        },
    )
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[add_tool]))
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="5")],
            isError=False,
        )
    )

    # First call: model asks to call add(2, 3); second call: model says "The result is 5."
    response_with_tool = MagicMock()
    response_with_tool.message = _make_message(
        "",
        tool_calls=[_make_tool_call("add", {"a": 2, "b": 3})],
    )
    response_final = MagicMock()
    response_final.message = _make_message("The result is 5.")

    with patch(
        "ollamacode.agent._ollama_chat_sync",
        side_effect=[response_with_tool, response_final],
    ):
        out = await run_agent_loop(
            session, "test-model", "What is 2+3?", max_tool_rounds=5
        )
    assert out == "The result is 5."
    session.call_tool.assert_awaited_once()
    call_args = session.call_tool.call_args
    assert call_args[0][0] == "add"
    assert call_args[0][1] == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_run_agent_loop_no_message_returns_error():
    """run_agent_loop returns 'No response from model.' when message is None."""
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[]))
    session.call_tool = AsyncMock()

    with patch("ollamacode.agent._ollama_chat_sync", return_value={}):
        out = await run_agent_loop(session, "test-model", "Hi")
    assert out == "No response from model."


@pytest.mark.asyncio
async def test_run_agent_loop_max_rounds_reached():
    """run_agent_loop returns max-rounds message when tool_calls every time."""
    add_tool = Tool(name="add", description="Add", inputSchema={"type": "object"})
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[add_tool]))
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="5")],
            isError=False,
        )
    )

    response_always_tool = MagicMock()
    response_always_tool.message = _make_message(
        "",
        tool_calls=[_make_tool_call("add", {"a": 1, "b": 2})],
    )

    with patch("ollamacode.agent._ollama_chat_sync", return_value=response_always_tool):
        out = await run_agent_loop(
            session, "test-model", "Add forever", max_tool_rounds=3
        )
    assert out == "(Max tool rounds reached; stopping.)"
    assert session.call_tool.await_count == 3  # type: ignore[attr-defined]
