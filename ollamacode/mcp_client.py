"""
MCP client wrapper: connect to one or more MCP servers and aggregate list_tools / call_tool.

Supports stdio, SSE, and Streamable HTTP transports. Single server via connect_mcp_stdio;
multiple servers via connect_mcp_servers (ClientSessionGroup).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator

from mcp import ClientSession, ClientSessionGroup, StdioServerParameters
from mcp.client.session_group import (
    ClientSessionParameters,
    SseServerParameters,
    StreamableHttpParameters,
)
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Implementation, ListToolsResult, TextContent

McpConnection = ClientSession | ClientSessionGroup


def _server_params_from_config(entry: dict[str, Any]) -> StdioServerParameters | SseServerParameters | StreamableHttpParameters:
    """Build MCP ServerParameters from a config dict (type, command/args or url)."""
    kind = (entry.get("type") or "stdio").lower()
    if kind == "stdio":
        return StdioServerParameters(
            command=entry.get("command", "python"),
            args=entry.get("args") or [],
            env=entry.get("env"),
        )
    if kind == "sse":
        return SseServerParameters(
            url=entry["url"],
            headers=entry.get("headers"),
            timeout=float(entry.get("timeout", 5)),
            sse_read_timeout=float(entry.get("sse_read_timeout", 60 * 5)),
        )
    if kind == "streamable_http":
        timeout = entry.get("timeout", 30)
        sse_read = entry.get("sse_read_timeout", 60 * 5)
        return StreamableHttpParameters(
            url=entry["url"],
            headers=entry.get("headers"),
            timeout=timedelta(seconds=timeout if isinstance(timeout, (int, float)) else 30),
            sse_read_timeout=timedelta(seconds=sse_read if isinstance(sse_read, (int, float)) else 60 * 5),
            terminate_on_close=entry.get("terminate_on_close", True),
        )
    raise ValueError(f"Unknown MCP server type: {kind}")


def _component_name_hook(name: str, server_info: Implementation) -> str:
    """Prefix tool names with server name to avoid collisions across servers."""
    return f"{server_info.name}_{name}"


@asynccontextmanager
async def connect_mcp_stdio(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to an MCP server over stdio. Yields a ClientSession."""
    params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def connect_mcp_servers(
    server_configs: list[dict[str, Any]],
) -> AsyncIterator[ClientSessionGroup]:
    """
    Connect to multiple MCP servers (stdio, sse, or streamable_http) and yield a ClientSessionGroup.

    Tool names are prefixed with the server's name (from initialize) to avoid collisions.
    """
    if not server_configs:
        raise ValueError("At least one MCP server config required")
    params_list = [_server_params_from_config(c) for c in server_configs]
    group = ClientSessionGroup(component_name_hook=_component_name_hook)
    async with group:
        for params in params_list:
            await group.connect_to_server(params, ClientSessionParameters())
        yield group


async def list_tools(connection: McpConnection) -> ListToolsResult:
    """List tools from the MCP server or session group."""
    if isinstance(connection, ClientSessionGroup):
        return ListToolsResult(tools=list(connection.tools.values()))
    return await connection.list_tools()


async def call_tool(
    connection: McpConnection,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> CallToolResult:
    """Call a tool by name with optional arguments."""
    args = arguments if arguments is not None else {}
    if isinstance(connection, ClientSessionGroup):
        return await connection.call_tool(name, args)
    return await connection.call_tool(name, args)


def tool_result_to_content(result: CallToolResult) -> str:
    """Extract a single string from CallToolResult for Ollama tool message."""
    if result.isError or not result.content:
        return str(result.content) if result.content else "Error: no content"
    parts = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts) if parts else ""
