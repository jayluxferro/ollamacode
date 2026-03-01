"""Unit tests for the agent loop (mocked Ollama and MCP)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from tests.helpers.mock_mcp import add_tool as make_tool, make_mock_mcp_session
from ollamacode.agent import (
    _filter_tools_by_policy,
    _format_tool_error_hint,
    _parse_tool_args,
    _tool_call_one_line,
    _truncate_messages,
    _truncate_tool_result,
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_one_tool,
    run_tools,
)


def test_filter_tools_by_policy():
    """_filter_tools_by_policy restricts by allowlist or blocklist."""
    tools = [
        {"function": {"name": "read_file", "description": "x"}},
        {"function": {"name": "run_command", "description": "y"}},
        {"function": {"name": "run_tests", "description": "z"}},
    ]
    assert len(_filter_tools_by_policy(tools, None, None)) == 3
    out = _filter_tools_by_policy(tools, ["read_file", "run_tests"], None)
    assert len(out) == 2
    assert {t["function"]["name"] for t in out} == {"read_file", "run_tests"}
    out = _filter_tools_by_policy(tools, None, ["run_command"])
    assert len(out) == 2
    assert {t["function"]["name"] for t in out} == {"read_file", "run_tests"}
    out = _filter_tools_by_policy(tools, ["read_file"], ["read_file"])
    assert len(out) == 0


def test_parse_tool_args_tolerates_extra_brace():
    """_parse_tool_args fixes common LLM mistake: extra '}' at end of JSON."""
    raw = '{"cwd":".","message":"Update after test pass"}}'
    out, err = _parse_tool_args(raw)
    assert err is None and out == {"cwd": ".", "message": "Update after test pass"}
    out, err = _parse_tool_args('{"a":1}}')
    assert err is None and out == {"a": 1}
    out, err = _parse_tool_args("{}")
    assert err is None and out == {}
    out, err = _parse_tool_args('{"x":1}')
    assert err is None and out == {"x": 1}


def test_parse_tool_args_tolerates_extra_bracket_and_newlines():
    """_parse_tool_args fixes extra ']' before '}' and unescaped newlines in strings."""
    out, err = _parse_tool_args('{"key": "val"]}')
    assert err is None and out == {"key": "val"}
    out, err = _parse_tool_args('{"a": "line1\n line2"}')
    assert err is None and out == {"a": "line1  line2"}


def test_parse_tool_args_returns_clear_error_on_invalid_json():
    """_parse_tool_args returns ({}, error_message) when JSON cannot be parsed."""
    out, err = _parse_tool_args("not json at all")
    assert out == {}
    assert err is not None
    assert "JSON" in err or "parse" in err.lower()
    out, err = _parse_tool_args("{invalid}")
    assert out == {}
    assert err is not None


def test_format_tool_error_hint_lookup():
    """_format_tool_error_hint returns hints from lookup table."""
    assert _format_tool_error_hint("") is None
    assert _format_tool_error_hint("FileNotFoundError: foo") is not None
    assert "File or path not found" in (
        _format_tool_error_hint("FileNotFoundError: foo") or ""
    )
    assert "Permission denied" in (_format_tool_error_hint("Permission denied") or "")
    assert "timed out" in (_format_tool_error_hint("request timed out") or "")
    assert "Command or executable" in (
        _format_tool_error_hint("command not found: pytest") or ""
    )
    assert "module not found" in (
        _format_tool_error_hint("ModuleNotFoundError: bar") or ""
    )
    assert _format_tool_error_hint("x" * 2500) is None


def test_truncate_tool_result_edge_cases():
    """_truncate_tool_result at and over limit."""
    short = "hello"
    assert _truncate_tool_result(short, 0) == short
    assert _truncate_tool_result(short, 10) == short
    assert _truncate_tool_result(short, 5) == short
    exact = "a" * 100
    assert _truncate_tool_result(exact, 100) == exact
    over = "a" * 150
    out = _truncate_tool_result(over, 100)
    assert len(out) > 100
    assert "truncated" in out
    assert out.startswith("a" * 100)


def test_truncate_messages_keeps_system_and_last_n():
    """_truncate_messages keeps system (if first) and last max_messages-1 messages."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
    ]
    out = _truncate_messages(messages, 3)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "You are helpful."
    assert len(out) == 3  # system + last 2 (max_messages - 1)
    assert [m["content"] for m in out[1:]] == ["3", "4"]
    out_no_sys = _truncate_messages(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], 1
    )
    assert len(out_no_sys) == 1
    assert out_no_sys[0]["content"] == "b"
    assert _truncate_messages(messages, 0) == messages
    assert _truncate_messages(messages, 10) == messages


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

    with patch(
        "ollamacode.agent.ollama_chat_async",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
        out = await run_agent_loop_no_mcp(
            "test-model", "Hi", system_prompt="You are helpful."
        )
    assert out == "Hello, world!"


@pytest.mark.asyncio
async def test_run_agent_loop_no_mcp_no_message():
    """run_agent_loop_no_mcp returns 'No response from model.' when message is None."""
    with patch(
        "ollamacode.agent.ollama_chat_async", new_callable=AsyncMock, return_value={}
    ):
        out = await run_agent_loop_no_mcp("test-model", "Hi")
    assert out == "No response from model."


@pytest.mark.asyncio
async def test_run_agent_loop_no_mcp_empty_content():
    """run_agent_loop_no_mcp returns stripped content (empty string if none)."""
    fake_response = MagicMock()
    fake_response.message = {"content": ""}

    with patch(
        "ollamacode.agent.ollama_chat_async",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
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

    with patch(
        "ollamacode.agent.ollama_chat_async",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
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
        "ollamacode.agent.ollama_chat_async",
        new_callable=AsyncMock,
        side_effect=[response_with_tool, response_final],
    ):
        out = await run_agent_loop(
            session, "test-model", "What is 2+3?", max_tool_rounds=5
        )
    assert out == "The result is 5."


@pytest.mark.asyncio
async def test_run_agent_loop_provider_gets_canonical_tools_and_system_hint():
    """Provider path gets canonical tool names and an injected tool-availability system hint."""
    tool = Tool(
        name="ollamacode-fs_read_file",
        description="Read file",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[tool]))
    session.call_tool = AsyncMock()

    provider = MagicMock()
    provider.chat_async = AsyncMock(return_value={"message": {"content": "ok"}})

    out = await run_agent_loop(
        session,
        "test-model",
        "read a file",
        provider=provider,
    )
    assert out == "ok"
    provider.chat_async.assert_awaited_once()
    args, _kwargs = provider.chat_async.call_args
    sent_messages = args[1]
    sent_tools = args[2]
    tool_names = [(t.get("function") or {}).get("name") for t in sent_tools]
    assert "read_file" in tool_names
    assert all(n and not str(n).startswith("functions::") for n in tool_names)
    assert any(
        (m.get("role") == "system" and "Tool availability:" in str(m.get("content")))
        for m in sent_messages
    )
    session.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_agent_loop_no_message_returns_error():
    """run_agent_loop returns 'No response from model.' when message is None."""
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[]))
    session.call_tool = AsyncMock()

    with patch(
        "ollamacode.agent.ollama_chat_async", new_callable=AsyncMock, return_value={}
    ):
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

    with patch(
        "ollamacode.agent.ollama_chat_async",
        new_callable=AsyncMock,
        return_value=response_always_tool,
    ):
        out = await run_agent_loop(
            session, "test-model", "Add forever", max_tool_rounds=3
        )
    # Doom loop detector catches identical tool calls before max_tool_rounds
    assert "Doom loop detected" in out or out == "(Max tool rounds reached; stopping.)"


@pytest.mark.asyncio
async def test_run_one_tool_returns_content_and_is_error():
    """run_one_tool returns (content, is_error, hint) from MCP call_tool."""
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="file contents")],
            isError=False,
        )
    )
    content, is_error, hint = await run_one_tool(
        session, "read_file", {"path": "foo.py"}, max_tool_result_chars=0
    )
    assert content == "file contents"
    assert is_error is False
    assert hint is None
    session.call_tool.assert_awaited_once_with("read_file", {"path": "foo.py"})


@pytest.mark.asyncio
async def test_run_tools_parallel_and_parse_error():
    """run_tools runs MCP tools in parallel and turns parse errors into synthetic results."""
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )
    items = [
        ("tool_a", {"x": 1}, None),
        ("tool_b", {}, "Invalid JSON at position 0"),
        ("tool_c", {"z": 2}, None),
    ]
    results = await run_tools(session, items, max_tool_result_chars=0)
    assert len(results) == 3
    name_a, args_a, content_a, is_err_a, hint_a = results[0]
    assert name_a == "tool_a" and content_a == "ok" and is_err_a is False
    name_b, args_b, content_b, is_err_b, hint_b = results[1]
    assert (
        name_b == "tool_b" and "could not be parsed" in content_b and is_err_b is True
    )
    name_c, args_c, content_c, is_err_c, hint_c = results[2]
    assert name_c == "tool_c" and content_c == "ok" and is_err_c is False
    assert session.call_tool.await_count == 2  # only tool_a and tool_c


@pytest.mark.asyncio
async def test_run_agent_loop_with_mock_mcp_server():
    """run_agent_loop works with the shared mock MCP session helper (list_tools + call_tool)."""
    session = make_mock_mcp_session(
        tools=[
            make_tool(
                "add",
                "Add two numbers",
                {"type": "object", "properties": {"a": {}, "b": {}}},
            )
        ],
        call_tool_return=CallToolResult(
            content=[TextContent(type="text", text="5")],
            isError=False,
        ),
    )
    response_with_tool = MagicMock()
    response_with_tool.message = _make_message(
        "", tool_calls=[_make_tool_call("add", {"a": 2, "b": 3})]
    )
    response_final = MagicMock()
    response_final.message = _make_message("The result is 5.")

    with patch(
        "ollamacode.agent.ollama_chat_async",
        new_callable=AsyncMock,
        side_effect=[response_with_tool, response_final],
    ):
        out = await run_agent_loop(
            session, "test-model", "What is 2+3?", max_tool_rounds=5
        )
    assert out == "The result is 5."
    session.list_tools.assert_awaited_once()
    session.call_tool.assert_awaited_once()
    assert session.call_tool.call_args[0][0] == "add"
    assert session.call_tool.call_args[0][1] == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_mock_mcp_session_call_tool_side_effect():
    """Mock MCP session supports call_tool_side_effect as callable (name, args) -> result."""

    def echo(name: str, args: dict) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text=f"{name}({args})")],
            isError=False,
        )

    session = make_mock_mcp_session(
        tools=[make_tool("echo")], call_tool_side_effect=echo
    )
    from ollamacode.mcp_client import call_tool

    result = await call_tool(session, "echo", {"x": 1})
    assert result.content
    text = (
        result.content[0].text
        if hasattr(result.content[0], "text")
        else str(result.content[0])
    )
    assert "echo" in text and "x" in text
