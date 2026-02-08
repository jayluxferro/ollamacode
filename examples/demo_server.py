"""
Minimal MCP server for testing OllamaCode: exposes add and echo tools over stdio.
Run: python -m ollamacode --mcp-command python --mcp-args examples/demo_server.py "What is 2+3?"
Or from repo root: uv run python examples/demo_server.py  (then connect with --mcp-args ...)
"""

from mcp.server.fastmcp import FastMCP  # requires mcp package with FastMCP (mcp 1.x)

mcp = FastMCP("OllamaCode Demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the given text."""
    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
