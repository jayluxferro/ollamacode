"""
Built-in reasoning MCP server: think tool for structured reasoning.

The model can call think(reasoning) to record a reasoning step; the tool returns OK.
Thinking is stored in a per-run buffer (last 10 entries). Optional: inject into context
in a later turn (handled by agent/caller if needed).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-reasoning")

# Module-level buffer: last N reasoning entries (simple; no cross-request persistence).
_THINK_BUFFER: list[str] = []
_THINK_BUFFER_MAX = 10


def get_think_buffer() -> list[str]:
    """Return current think buffer (for optional injection into context)."""
    return list(_THINK_BUFFER)


def clear_think_buffer() -> None:
    """Clear the think buffer (e.g. at start of new turn or session)."""
    global _THINK_BUFFER
    _THINK_BUFFER.clear()


@mcp.tool()
def think(reasoning: str) -> str:
    """Record a reasoning or thinking step. Call this before answering to work through a problem. Returns OK. You can call it multiple times in one turn."""
    global _THINK_BUFFER
    s = (reasoning or "").strip()
    if s:
        _THINK_BUFFER.append(s)
        if len(_THINK_BUFFER) > _THINK_BUFFER_MAX:
            _THINK_BUFFER.pop(0)
    return "OK"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
