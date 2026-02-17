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
    """Write content to a file. Path is relative to workspace root (OLLAMACODE_FS_ROOT or cwd). Creates parent dirs if needed."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path} (absolute: {p})"


@mcp.tool()
def list_dir(path: str = ".") -> list[str]:
    """List directory entries (files and directories). Path is relative to workspace root. Default '.'."""
    p = _resolve(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return sorted(e.name for e in p.iterdir())


@mcp.tool()
def edit_file(
    path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """Surgical edit: replace old_string with new_string in file. Path relative to workspace root. replace_all: replace every occurrence (default: first only)."""
    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or not found: {path}")
    text = p.read_text(encoding="utf-8", errors="replace")
    if replace_all:
        if old_string not in text:
            return f"No occurrence of old_string in {path}"
        new_text = text.replace(old_string, new_string)
    else:
        if old_string not in text:
            return f"No occurrence of old_string in {path}"
        new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    return (
        f"Edited {path} (1 replacement)"
        if not replace_all
        else f"Edited {path} (all replacements)"
    )


@mcp.tool()
def multi_edit(edits: list[dict]) -> str:
    """Batch edits in one call. edits: list of {path, old_string?, new_string}. If old_string omitted, whole file is replaced with new_string. Paths relative to workspace root."""
    results: list[str] = []
    for i, item in enumerate(edits):
        if not isinstance(item, dict):
            results.append(f"[{i}] skip: not a dict")
            continue
        path_val = item.get("path")
        new_text = item.get("newText") or item.get("new_string")
        old_str = item.get("oldText") or item.get("old_string")
        if path_val is None:
            results.append(f"[{i}] skip: missing path")
            continue
        if new_text is None:
            results.append(f"[{i}] {path_val}: skip: missing newText/new_string")
            continue
        try:
            p = _resolve(str(path_val))
            if not p.is_file():
                results.append(f"[{i}] {path_val}: file not found")
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if old_str is None or old_str == "":
                p.write_text(
                    new_text if isinstance(new_text, str) else str(new_text),
                    encoding="utf-8",
                )
                results.append(f"[{i}] {path_val}: overwrote")
            else:
                old_s = old_str if isinstance(old_str, str) else str(old_str)
                new_s = new_text if isinstance(new_text, str) else str(new_text)
                if old_s not in text:
                    results.append(f"[{i}] {path_val}: old_string not found")
                    continue
                new_content = text.replace(old_s, new_s, 1)
                p.write_text(new_content, encoding="utf-8")
                results.append(f"[{i}] {path_val}: replaced")
        except Exception as e:
            results.append(f"[{i}] {path_val}: {e}")
    return "\n".join(results)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
