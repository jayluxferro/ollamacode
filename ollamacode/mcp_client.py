"""
MCP client wrapper: connect to one or more MCP servers and aggregate list_tools / call_tool.

Supports stdio, SSE, and Streamable HTTP transports. Single server via connect_mcp_stdio;
multiple servers via connect_mcp_servers (ClientSessionGroup).

Plugins: register additional MCP server types via entry point group ``ollamacode.mcp_server_types``.
Each entry point name is the config ``type`` value (e.g. ``stdio``, ``sse``, ``streamable_http``);
the entry point must be a callable ``(entry: dict) -> StdioServerParameters | SseServerParameters | StreamableHttpParameters``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import contextvars
from datetime import timedelta
import logging
import os
import shlex
from typing import Any, Callable, AsyncIterator, Dict, List
from urllib.parse import urlparse

from mcp import ClientSession, ClientSessionGroup, StdioServerParameters
from mcp.client.session_group import (
    ClientSessionParameters,
    SseServerParameters,
    StreamableHttpParameters,
)
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Implementation, ListToolsResult, TextContent

McpConnection = ClientSession | ClientSessionGroup

# Type of MCP server params returned by built-in and plugin builders.
ServerParamsType = (
    StdioServerParameters | SseServerParameters | StreamableHttpParameters
)

# Entry point group for plugins that add MCP server types.
MCP_SERVER_TYPES_ENTRY_POINT_GROUP = "ollamacode.mcp_server_types"

# Tool-call guard: set True to hard-block any tool calls (planner/verify phases).
_TOOL_CALLS_DISABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ollamacode_tool_calls_disabled", default=False
)


def disable_tool_calls() -> contextvars.Token:
    """Disable tool calls in the current context; returns a token for reset()."""
    return _TOOL_CALLS_DISABLED.set(True)


def reset_tool_calls(token: contextvars.Token) -> None:
    """Reset tool-call guard to previous state."""
    _TOOL_CALLS_DISABLED.reset(token)


def _builtin_stdio_params(entry: Dict[str, Any]) -> StdioServerParameters:
    merged_env = dict(entry.get("env") or {})
    merged_env.setdefault("OLLAMACODE_MCP_SERVER_LOG_LEVEL", "ERROR")
    merged_env.setdefault("OLLAMACODE_MCP_STDERR_QUIET", "1")
    cmd = entry.get("command", "python")
    args = entry.get("args") or []
    quiet_stderr = merged_env.get(
        "OLLAMACODE_MCP_STDERR_QUIET", "1"
    ).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if quiet_stderr and os.name == "posix":
        quoted = " ".join(shlex.quote(str(p)) for p in [cmd, *args])
        shell_cmd = f"exec {quoted} 2>/dev/null"
        return StdioServerParameters(
            command="sh",
            args=["-lc", shell_cmd],
            env=merged_env or None,
        )
    return StdioServerParameters(
        command=cmd,
        args=args,
        env=merged_env or None,
    )


logger = logging.getLogger(__name__)

_MAX_MCP_URL_LEN = 2048
_ALLOWED_MCP_URL_SCHEMES = {"http", "https"}


def _validate_mcp_url(url: str, context: str = "MCP server") -> None:
    """Validate a URL for MCP SSE/HTTP connections. Raises ValueError on invalid input."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{context}: url is required and must be a non-empty string")
    if len(url) > _MAX_MCP_URL_LEN:
        raise ValueError(
            f"{context}: url too long ({len(url)} chars, max {_MAX_MCP_URL_LEN})"
        )
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"{context}: invalid url: {exc}") from exc
    if parsed.scheme not in _ALLOWED_MCP_URL_SCHEMES:
        raise ValueError(
            f"{context}: url scheme must be http or https, got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ValueError(f"{context}: url must include a hostname")


def _builtin_sse_params(entry: Dict[str, Any]) -> SseServerParameters:
    url = entry.get("url") or ""
    _validate_mcp_url(url, context="SSE MCP server")
    timeout = entry.get("timeout", 5)
    sse_read = entry.get("sse_read_timeout", 60 * 5)
    try:
        timeout = max(1.0, min(float(timeout), 300.0))
    except (TypeError, ValueError):
        timeout = 5.0
    try:
        sse_read = max(1.0, min(float(sse_read), 3600.0))
    except (TypeError, ValueError):
        sse_read = 300.0
    return SseServerParameters(
        url=url,
        headers=entry.get("headers"),
        timeout=timeout,
        sse_read_timeout=sse_read,
    )


def _builtin_streamable_http_params(
    entry: Dict[str, Any],
) -> StreamableHttpParameters:
    url = entry.get("url") or ""
    _validate_mcp_url(url, context="Streamable HTTP MCP server")
    timeout = entry.get("timeout", 30)
    sse_read = entry.get("sse_read_timeout", 60 * 5)
    try:
        timeout_s = max(1.0, min(float(timeout), 300.0))
    except (TypeError, ValueError):
        timeout_s = 30.0
    try:
        sse_read_s = max(1.0, min(float(sse_read), 3600.0))
    except (TypeError, ValueError):
        sse_read_s = 300.0
    return StreamableHttpParameters(
        url=url,
        headers=entry.get("headers"),
        timeout=timedelta(seconds=timeout_s),
        sse_read_timeout=timedelta(seconds=sse_read_s),
        terminate_on_close=entry.get("terminate_on_close", True),
    )


_BUILTIN_SERVER_TYPES: Dict[str, Callable[[Dict[str, Any]], ServerParamsType]] = {
    "stdio": _builtin_stdio_params,
    "sse": _builtin_sse_params,
    "streamable_http": _builtin_streamable_http_params,
}


def _get_server_type_registry() -> Dict[
    str, Callable[[Dict[str, Any]], ServerParamsType]
]:
    """Merge built-in MCP server types with plugins from entry points."""
    registry = dict(_BUILTIN_SERVER_TYPES)
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=MCP_SERVER_TYPES_ENTRY_POINT_GROUP)
    except Exception as exc:
        logger.warning("Failed to load MCP server type entry points: %s", exc)
        return registry
    for ep in eps:
        try:
            registry[ep.name.lower()] = ep.load()
        except Exception as exc:
            logger.warning("Failed to load MCP server type plugin %r: %s", ep.name, exc)
            continue
    return registry


def get_registered_mcp_server_types() -> List[str]:
    """Return sorted list of registered MCP server type names (built-in + plugins)."""
    return sorted(_get_server_type_registry().keys())


def _server_params_from_config(
    entry: Dict[str, Any],
) -> StdioServerParameters | SseServerParameters | StreamableHttpParameters:
    """Build MCP ServerParameters from a config dict (type, command/args or url). Uses built-in types and plugins from entry point group ollamacode.mcp_server_types."""
    kind = (entry.get("type") or "stdio").lower()
    registry = _get_server_type_registry()
    if kind not in registry:
        raise ValueError(
            f"Unknown MCP server type: {kind!r}. Registered: {', '.join(sorted(registry.keys()))}"
        )
    return registry[kind](entry)


def _component_name_hook(name: str, server_info: Implementation) -> str:
    """Prefix tool names with server name to avoid collisions across servers."""
    return f"{server_info.name}_{name}"


@asynccontextmanager
async def connect_mcp_stdio(
    command: str,
    args: List[str] | None = None,
    env: Dict[str, str] | None = None,
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
    server_configs: List[Dict[str, Any]],
) -> AsyncIterator[ClientSessionGroup]:
    """Connect to multiple MCP servers (stdio, sse, or streamable_http) and yield a ClientSessionGroup.

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


# Per-session tool-list cache for single ClientSession connections.
# ClientSessionGroup already caches in connection.tools; this covers the single-server case.
_single_session_tools_cache: Dict[int, ListToolsResult] = {}


def invalidate_tools_cache(connection: McpConnection) -> None:
    """Remove cached tool list for a connection (call on reconnect or tool change)."""
    _single_session_tools_cache.pop(id(connection), None)


async def list_tools(connection: McpConnection) -> ListToolsResult:
    """List tools from the MCP server or session group.

    For ClientSessionGroup the result is already O(1) from the group's internal cache.
    For a single ClientSession the result is cached after the first call so subsequent
    agent turns don't make a redundant network round-trip.
    """
    if isinstance(connection, ClientSessionGroup):
        return ListToolsResult(tools=list(connection.tools.values()))
    conn_id = id(connection)
    if conn_id in _single_session_tools_cache:
        return _single_session_tools_cache[conn_id]
    result = await connection.list_tools()
    _single_session_tools_cache[conn_id] = result
    return result


# Model name -> our tool name. Add aliases so Ollama's harmony parser has a mapping (avoids "no reverse mapping" warning).
TOOL_NAME_ALIASES: Dict[str, str] = {
    "open_file": "read_file",
    "container.exec": "run_command",
}

# Ollama's harmony parser may return tool names as "functions::<name>"; strip so we resolve to the real tool.
HARMONY_FUNCTIONS_PREFIX = "functions::"


def _normalize_tool_name(name: str) -> str:
    """Strip functions:: prefix and apply aliases. Does not check against any tool list."""
    lookup = name
    if lookup.startswith(HARMONY_FUNCTIONS_PREFIX):
        lookup = lookup[len(HARMONY_FUNCTIONS_PREFIX) :]
    return TOOL_NAME_ALIASES.get(lookup, lookup)


def _resolve_tool_name_for_tools(tools: Dict[str, Any], name: str) -> str:
    """Resolve name against a tools dict (exact, then normalized, then suffix). Raises KeyError if not found."""
    if name in tools:
        return name
    lookup = _normalize_tool_name(name)
    candidates = [k for k in tools if k == lookup or k.endswith("_" + lookup)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[0]
    raise KeyError(f"Unknown tool: {name!r}. Available: {list(tools.keys())}")


def get_tool_name(connection: McpConnection, name: str) -> str:
    """
    Resolve a tool name for use with this connection.
    Strips ``functions::`` prefix and applies aliases (e.g. open_file -> read_file).
    For a session group, also resolves to the prefixed name (e.g. read_file -> ollamacode-fs_read_file).
    Raises KeyError if the name cannot be resolved (group only; single session returns normalized name).
    """
    if isinstance(connection, ClientSessionGroup):
        return _resolve_tool_name_for_tools(connection.tools, name)
    return _normalize_tool_name(name)


def _resolve_tool_name(group: ClientSessionGroup, name: str) -> str:
    """Resolve tool name for a group. Use get_tool_name(connection, name) in new code."""
    return get_tool_name(group, name)


def _unknown_tool_result(name: str, available: List[str]) -> CallToolResult:
    """Return a CallToolResult for an unknown tool so the model can retry with a valid tool."""
    short = ["_".join(k.split("_", 1)[1:]) if "_" in k else k for k in available]
    hint = (
        "Use read_file, edit_file, multi_edit, or apply_patch to modify files."
        if "apply_patch" in name or "patch" in name.lower()
        else ""
    )
    msg = f"Tool {name!r} is not available. Available tools: {', '.join(sorted(set(short)))}. {hint}".strip()
    return CallToolResult(content=[TextContent(type="text", text=msg)], isError=True)


async def call_tool(
    connection: McpConnection,
    name: str,
    arguments: Dict[str, Any] | None = None,
) -> CallToolResult:
    """Call a tool by name with optional arguments. Uses get_tool_name for resolution."""
    if _TOOL_CALLS_DISABLED.get():
        return CallToolResult(
            content=[
                TextContent(type="text", text="Tool calls are disabled for this phase.")
            ],
            isError=True,
        )
    args = arguments or {}
    if isinstance(connection, ClientSessionGroup):
        try:
            resolved = get_tool_name(connection, name)
        except KeyError:
            return _unknown_tool_result(name, list(connection.tools.keys()))
        return await connection.call_tool(resolved, args)
    resolved = get_tool_name(connection, name)
    return await connection.call_tool(resolved, args)


def tool_result_to_content(result: CallToolResult) -> str:
    """Extract a single string from CallToolResult for Ollama tool message."""
    if not isinstance(result, CallToolResult):
        logger.warning(
            "tool_result_to_content received unexpected type %s; converting to string",
            type(result).__name__,
        )
        return str(result) if result else "Error: no content"
    if result.isError or not result.content:
        return str(result.content) if result.content else "Error: no content"
    if not isinstance(result.content, (list, tuple)):
        logger.warning(
            "CallToolResult.content is %s, expected list; converting to string",
            type(result.content).__name__,
        )
        return str(result.content)
    parts: List[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts) if parts else ""
