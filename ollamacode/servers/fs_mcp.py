"""
Built-in filesystem MCP server: read_file, write_file, list_dir (workspace-scoped).

Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-fs")


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _resolve(path: str) -> Path:
    p = _root() / path.lstrip("/")
    resolved = p.resolve()
    if not resolved.is_relative_to(_root()):
        raise ValueError(f"Path {path!r} is outside workspace root {_root()}")
    return resolved


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file. Path is relative to workspace root (OLLAMACODE_FS_ROOT or cwd)."""
    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. Path is relative to workspace root. Creates parent dirs if needed."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


@mcp.tool()
def list_dir(path: str = ".") -> list[str]:
    """List directory entries (files and directories). Path is relative to workspace root. Default '.'."""
    p = _resolve(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return sorted(e.name for e in p.iterdir())


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
