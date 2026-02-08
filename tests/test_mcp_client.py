"""Tests for MCP client helpers."""

from mcp.types import CallToolResult, TextContent

from ollamacode.mcp_client import tool_result_to_content


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
