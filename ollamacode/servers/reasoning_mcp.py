"""
Built-in reasoning MCP server: think tool for structured reasoning.

The model can call think(reasoning) to record a reasoning step; the tool returns OK.
Thinking is stored in a per-session buffer (last 10 entries). Optional: inject into context
in a later turn (handled by agent/caller if needed).
"""

import os

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()

mcp = FastMCP("ollamacode-reasoning")

# Per-session buffers: keyed by session ID. Falls back to a default "" key
# when OLLAMACODE_SESSION_ID is not set.
_THINK_BUFFERS: dict[str, list[str]] = {}
_THINK_BUFFER_MAX = 10


def _session_id() -> str:
    """Return current session ID from env, or empty string as default."""
    return os.environ.get("OLLAMACODE_SESSION_ID", "")


def get_think_buffer(session_id: str | None = None) -> list[str]:
    """Return current think buffer for the given session (for optional injection into context)."""
    sid = session_id if session_id is not None else _session_id()
    return list(_THINK_BUFFERS.get(sid, []))


def clear_think_buffer(session_id: str | None = None) -> None:
    """Clear the think buffer for the given session (e.g. at start of new turn or session)."""
    sid = session_id if session_id is not None else _session_id()
    if sid in _THINK_BUFFERS:
        _THINK_BUFFERS[sid].clear()


@mcp.tool()
def think(reasoning: str) -> str:
    """Record a reasoning or thinking step. Call this before answering to work through a problem. Returns OK. You can call it multiple times in one turn."""
    sid = _session_id()
    s = (reasoning or "").strip()
    if s:
        buf = _THINK_BUFFERS.setdefault(sid, [])
        buf.append(s)
        if len(buf) > _THINK_BUFFER_MAX:
            buf.pop(0)
    return "OK"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
