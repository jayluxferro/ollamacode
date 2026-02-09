"""Tests for MCP → Ollama tool bridge."""

from mcp.types import Tool

from ollamacode.bridge import (
    mcp_tool_to_ollama,
    mcp_tools_to_ollama,
    use_short_names_for_builtin_tools,
)


def test_mcp_tool_to_ollama():
    """Single MCP tool converts to Ollama function format."""
    mcp_tool = Tool(
        name="add",
        description="Add two numbers",
        inputSchema={
            "type": "object",
            "required": ["a", "b"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    )
    out = mcp_tool_to_ollama(mcp_tool)
    assert out["type"] == "function"
    assert out["function"]["name"] == "add"
    assert out["function"]["description"] == "Add two numbers"
    assert out["function"]["parameters"]["required"] == ["a", "b"]


def test_mcp_tool_to_ollama_empty_description():
    """None/empty description becomes empty string."""
    mcp_tool = Tool(name="echo", description=None, inputSchema={"type": "object"})
    out = mcp_tool_to_ollama(mcp_tool)
    assert out["function"]["description"] == ""


def test_mcp_tools_to_ollama():
    """List of MCP tools converts to list of Ollama tools."""
    tools = [
        Tool(name="add", description="Add", inputSchema={"type": "object"}),
        Tool(name="echo", description="Echo", inputSchema={"type": "object"}),
    ]
    out = mcp_tools_to_ollama(tools)
    assert len(out) == 2
    assert out[0]["function"]["name"] == "add"
    assert out[1]["function"]["name"] == "echo"


def test_mcp_tools_to_ollama_empty():
    """Empty list returns empty list."""
    assert mcp_tools_to_ollama([]) == []


def test_use_short_names_for_builtin_tools():
    """Built-in server tool names are rewritten to short form (read_file, run_command, etc.)."""
    ollama_tools = [
        {
            "type": "function",
            "function": {
                "name": "ollamacode-fs_read_file",
                "description": "Read file",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "BURP_REST_API_scan",
                "description": "Scan",
                "parameters": {},
            },
        },
    ]
    out = use_short_names_for_builtin_tools(ollama_tools)
    assert out[0]["function"]["name"] == "read_file"
    assert out[1]["function"]["name"] == "BURP_REST_API_scan"
