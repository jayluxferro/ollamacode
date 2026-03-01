"""Tests for MCP client helpers."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.client.session_group import SseServerParameters
from mcp.types import CallToolResult, ListToolsResult, TextContent

from ollamacode.mcp_client import (
    MCP_SERVER_TYPES_ENTRY_POINT_GROUP,
    _resolve_tool_name_for_tools,
    _server_params_from_config,
    get_registered_mcp_server_types,
    get_tool_name,
    list_tools,
    tool_result_to_content,
)


def test_tool_result_to_content_single_text():
    """Single TextContent block returns its text."""
    result = CallToolResult(
        content=[TextContent(type="text", text="hello")],
        isError=False,
    )
    assert tool_result_to_content(result) == "hello"


def test_tool_result_to_content_multiple_text():
    """Multiple text blocks are joined with newlines."""
    result = CallToolResult(
        content=[
            TextContent(type="text", text="line1"),
            TextContent(type="text", text="line2"),
        ],
        isError=False,
    )
    assert tool_result_to_content(result) == "line1\nline2"


def test_tool_result_to_content_empty():
    """Empty content returns 'Error: no content' when isError or empty."""
    result = CallToolResult(content=[], isError=False)
    assert tool_result_to_content(result) == "Error: no content"


def test_tool_result_to_content_error():
    """Error result is handled (content still extracted if present)."""
    result = CallToolResult(
        content=[TextContent(type="text", text="failed")],
        isError=True,
    )
    # Current impl returns str(content) for error; we just check it doesn't crash
    out = tool_result_to_content(result)
    assert isinstance(out, str)


def test_get_tool_name_single_session_normalizes():
    """Single session (non-Group): strips functions:: and applies alias."""
    session = object()  # not ClientSessionGroup
    assert get_tool_name(session, "functions::read_file") == "read_file"
    assert get_tool_name(session, "open_file") == "read_file"
    assert get_tool_name(session, "read_file") == "read_file"


def test_resolve_tool_name_for_tools_exact_and_suffix():
    """Group resolution (exact, suffix, prefix strip, alias) used by get_tool_name."""
    tools = {"ollamacode-fs_read_file": None, "other_tool": None}
    assert (
        _resolve_tool_name_for_tools(tools, "ollamacode-fs_read_file")
        == "ollamacode-fs_read_file"
    )
    assert _resolve_tool_name_for_tools(tools, "read_file") == "ollamacode-fs_read_file"
    assert (
        _resolve_tool_name_for_tools(tools, "functions::read_file")
        == "ollamacode-fs_read_file"
    )
    assert _resolve_tool_name_for_tools(tools, "open_file") == "ollamacode-fs_read_file"
    assert _resolve_tool_name_for_tools(tools, "other_tool") == "other_tool"


def test_resolve_tool_name_for_tools_unknown_raises():
    """Group resolution raises KeyError with available list when tool unknown."""
    tools = {"only_tool": None}
    try:
        _resolve_tool_name_for_tools(tools, "nonexistent")
    except KeyError as e:
        assert "nonexistent" in str(e)
        assert "only_tool" in str(e)
    else:
        raise AssertionError("Expected KeyError")


def test_server_params_from_config_stdio_default():
    """_server_params_from_config returns StdioServerParameters for stdio with defaults."""
    params = _server_params_from_config({"type": "stdio"})
    if os.name == "posix":
        assert params.command == "sh"
        assert params.args and params.args[0] == "-lc"
        assert "python" in params.args[1]
        assert "2>/dev/null" in params.args[1]
    else:
        assert params.command == "python"
        assert params.args == []
    assert params.env == {
        "OLLAMACODE_MCP_SERVER_LOG_LEVEL": "ERROR",
        "OLLAMACODE_MCP_STDERR_QUIET": "1",
    }


def test_server_params_from_config_stdio_with_args_and_env():
    """_server_params_from_config passes command, args, env for stdio."""
    params = _server_params_from_config(
        {
            "type": "stdio",
            "command": "/usr/bin/python3",
            "args": ["-m", "server"],
            "env": {"FOO": "bar"},
        }
    )
    if os.name == "posix":
        assert params.command == "sh"
        assert params.args and params.args[0] == "-lc"
        assert "/usr/bin/python3" in params.args[1]
        assert "-m server" in params.args[1]
    else:
        assert params.command == "/usr/bin/python3"
        assert params.args == ["-m", "server"]
    assert params.env == {
        "FOO": "bar",
        "OLLAMACODE_MCP_SERVER_LOG_LEVEL": "ERROR",
        "OLLAMACODE_MCP_STDERR_QUIET": "1",
    }


def test_server_params_from_config_sse():
    """_server_params_from_config returns SseServerParameters for sse."""
    params = _server_params_from_config(
        {
            "type": "sse",
            "url": "http://localhost:8080/sse",
            "timeout": 10,
        }
    )
    assert params.url == "http://localhost:8080/sse"
    assert params.timeout == 10.0


def test_server_params_from_config_streamable_http():
    """_server_params_from_config returns StreamableHttpParameters for streamable_http."""
    params = _server_params_from_config(
        {
            "type": "streamable_http",
            "url": "http://localhost:8080/stream",
        }
    )
    assert params.url == "http://localhost:8080/stream"


def test_server_params_from_config_unknown_raises():
    """_server_params_from_config raises ValueError for unknown type."""
    with pytest.raises(ValueError) as exc_info:
        _server_params_from_config({"type": "unknown"})
    assert "Unknown MCP server type" in str(exc_info.value)
    assert "Registered:" in str(exc_info.value)


def test_get_registered_mcp_server_types_includes_builtins():
    """get_registered_mcp_server_types returns at least stdio, sse, streamable_http."""
    types = get_registered_mcp_server_types()
    assert "stdio" in types
    assert "sse" in types
    assert "streamable_http" in types
    assert types == sorted(types)


def test_mcp_server_types_entry_point_group_constant():
    """Entry point group name is defined for plugins."""
    assert MCP_SERVER_TYPES_ENTRY_POINT_GROUP == "ollamacode.mcp_server_types"


def test_plugin_mcp_server_type_used_when_registered():
    """A plugin registered via entry_points is used by get_registered_mcp_server_types and _server_params_from_config."""

    def custom_builder(entry):
        return SseServerParameters(
            url=entry.get("url", "http://custom"),
            headers=entry.get("headers"),
            timeout=float(entry.get("timeout", 5)),
            sse_read_timeout=float(entry.get("sse_read_timeout", 300)),
        )

    ep = MagicMock()
    ep.name = "custom"
    ep.load = MagicMock(return_value=custom_builder)

    def fake_entry_points(*, group):
        if group != MCP_SERVER_TYPES_ENTRY_POINT_GROUP:
            return []
        return [ep]

    with patch("importlib.metadata.entry_points", fake_entry_points):
        types = get_registered_mcp_server_types()
        assert "custom" in types
        params = _server_params_from_config(
            {"type": "custom", "url": "http://example.com"}
        )
        assert isinstance(params, SseServerParameters)
        assert params.url == "http://example.com"


@pytest.mark.asyncio
async def test_list_tools_session_calls_list_tools():
    """list_tools with single session awaits connection.list_tools()."""
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[]))
    result = await list_tools(session)
    assert result.tools == []
    session.list_tools.assert_awaited_once()
