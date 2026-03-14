"""Built-in interactive question MCP server."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import configure_server_logging

configure_server_logging()

mcp = FastMCP("ollamacode-question")


@mcp.tool()
def question(questions: list[dict[str, Any]]) -> str:
    """Ask the user one or more structured questions. Interactive clients intercept this tool."""
    return (
        "Question tool requires an interactive OllamaCode client. "
        "If you see this message, the client did not intercept the question request."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
