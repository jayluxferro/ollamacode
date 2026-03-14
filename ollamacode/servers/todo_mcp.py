"""Built-in session todo MCP server: todoread and todowrite."""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import configure_server_logging
from ..sessions import load_session_todos, save_session_todos

configure_server_logging()

mcp = FastMCP("ollamacode-todo")


def _require_session_id() -> str:
    session_id = os.environ.get("OLLAMACODE_SESSION_ID", "").strip()
    if not session_id:
        raise ValueError(
            "No active session id available for todo tools. Start or resume a session first."
        )
    return session_id


@mcp.tool()
def todoread() -> str:
    """Read the current session todo list."""
    session_id = _require_session_id()
    todos = load_session_todos(session_id) or []
    return json.dumps(todos, indent=2, ensure_ascii=False)


@mcp.tool()
def todowrite(todos: list[dict[str, Any]]) -> str:
    """Replace the current session todo list with the provided ordered list."""
    session_id = _require_session_id()
    save_session_todos(session_id, todos)
    saved = load_session_todos(session_id) or []
    return json.dumps(saved, indent=2, ensure_ascii=False)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
