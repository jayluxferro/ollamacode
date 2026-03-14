"""Built-in interactive task delegation MCP server."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import configure_server_logging

configure_server_logging()

mcp = FastMCP("ollamacode-task")


@mcp.tool()
def task(
    description: str,
    prompt: str,
    subagent_type: str,
    task_id: str | None = None,
    command: str | None = None,
) -> str:
    """Delegate work to a configured subagent. Interactive clients intercept this tool."""
    return (
        "Task delegation requires an interactive OllamaCode client. "
        "If you see this message, the client did not intercept the task request."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
