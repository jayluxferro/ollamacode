"""
Built-in state MCP server: get_state, update_state, clear_state, append_recent_file.

State is stored in ~/.ollamacode/state.json (recent files, preferences). Lets the assistant remember context across sessions.
"""

from mcp.server.fastmcp import FastMCP

from ..state import append_recent_file as _append_recent_file
from ..state import clear_state as _clear_state
from ..state import get_state as _get_state
from ..state import update_state as _update_state

mcp = FastMCP("ollamacode-state")


@mcp.tool()
def get_state() -> dict:
    """Return persistent state (recent_files, preferences, etc.) from ~/.ollamacode/state.json."""
    return _get_state()


@mcp.tool()
def update_state(
    recent_files: list[str] | None = None,
    preferences: dict | None = None,
) -> str:
    """Update state. recent_files: list of paths to remember. preferences: dict of key-value preferences to merge."""
    return _update_state(recent_files=recent_files, preferences=preferences)


@mcp.tool()
def append_recent_file(path: str) -> str:
    """Record a file path as recently used (for context across sessions)."""
    return _append_recent_file(path)


@mcp.tool()
def clear_state() -> str:
    """Clear all persistent state (recent files, preferences)."""
    return _clear_state()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
