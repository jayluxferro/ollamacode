"""
Built-in skills & memory MCP server: list_skills, read_skill, write_skill, save_memory.

Skills are Markdown files in ~/.ollamacode/skills (global) and .ollamacode/skills (workspace).
Workspace root: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

from ..skills import (
    list_skills as _list_skills,
    read_skill as _read_skill,
    save_memory as _save_memory,
    write_skill as _write_skill,
)

configure_server_logging()

mcp = FastMCP("ollamacode-skills")


def _workspace_root() -> str | None:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return os.path.abspath(root) if root else os.getcwd()


@mcp.tool()
def list_skills() -> list[str]:
    """List available skill names (from global and workspace skills directories)."""
    return _list_skills(_workspace_root())


@mcp.tool()
def read_skill(name: str) -> str:
    """Read a skill by name. Returns its content or an error message if not found or invalid name."""
    content = _read_skill(name, _workspace_root())
    if content is None:
        return f"Skill '{name}' not found or invalid name (use letters, numbers, underscore, hyphen)."
    return content


@mcp.tool()
def write_skill(name: str, content: str) -> str:
    """Create or overwrite a skill. Name: letters, numbers, underscore, hyphen only. Content: Markdown text."""
    return _write_skill(name, content, _workspace_root())


@mcp.tool()
def save_memory(key: str, value: str) -> str:
    """Save a fact or preference to persistent memory (appends to the 'memory' skill). Use for things to remember across sessions."""
    return _save_memory(key, value, _workspace_root())


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
