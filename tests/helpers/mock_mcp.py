"""
Mock MCP server for agent and integration tests.

Provides a session-like object that implements list_tools and call_tool
without starting a real MCP server process.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool


def make_mock_mcp_session(
    tools: list[Tool] | None = None,
    *,
    call_tool_return: CallToolResult | None = None,
    call_tool_side_effect: list[CallToolResult]
    | Callable[[str, dict[str, Any]], CallToolResult]
    | None = None,
) -> MagicMock:
    """
    Build a mock MCP connection (session) for use in run_agent_loop / list_tools / call_tool tests.

    Args:
        tools: Tools to return from list_tools(). Defaults to [].
        call_tool_return: Single result for every call_tool(name, args). Ignored if call_tool_side_effect is set.
        call_tool_side_effect: Either a list of CallToolResult (one per call) or a callable (name, args) -> CallToolResult.

    Returns:
        A MagicMock with list_tools and call_tool configured. Use with run_agent_loop(session, ...)
        or with ollamacode.mcp_client.list_tools(session) / call_tool(session, name, args).
    """
    tools = tools or []
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=ListToolsResult(tools=tools))

    if call_tool_side_effect is not None:
        if callable(call_tool_side_effect):

            async def _call(name: str, args: dict[str, Any]) -> CallToolResult:
                return call_tool_side_effect(name, args or {})

            session.call_tool = AsyncMock(side_effect=_call)
        else:
            session.call_tool = AsyncMock(side_effect=list(call_tool_side_effect))
    else:
        default = call_tool_return or CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
        session.call_tool = AsyncMock(return_value=default)

    return session


def add_tool(
    name: str, description: str = "", input_schema: dict[str, Any] | None = None
) -> Tool:
    """Convenience: build a Tool for use with make_mock_mcp_session(tools=[...])."""
    return Tool(
        name=name,
        description=description or name,
        inputSchema=input_schema or {"type": "object", "properties": {}},
    )
