"""
Built-in filesystem MCP server: read_file, write_file, list_dir (workspace-scoped).

Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
Sandbox: OLLAMACODE_SANDBOX_LEVEL controls access (readonly/supervised/full).
"""

import os
import difflib
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-fs")


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _resolve(path: str, *, allow_write: bool = False) -> Path:
    """Resolve *path* relative to workspace root and enforce sandbox policy."""
    from ollamacode.sandbox import check_fs_path

    workspace = _root()
    check_fs_path(path, workspace, allow_write=allow_write)
    p = workspace / path.lstrip("/")
    resolved = p.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Path {path!r} is outside workspace root {workspace}")
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
    p = _resolve(path, allow_write=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1":
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                content.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        return "Dry run (no write).\n" + (diff or "(no changes)")
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
    p = _resolve(path, allow_write=True)
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
    if os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1":
        diff = "\n".join(
            difflib.unified_diff(
                text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        return "Dry run (no write).\n" + (diff or "(no changes)")
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
            p = _resolve(str(path_val), allow_write=True)
            if not p.is_file():
                results.append(f"[{i}] {path_val}: file not found")
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if old_str is None or old_str == "":
                new_val = new_text if isinstance(new_text, str) else str(new_text)
                if os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1":
                    diff = "\n".join(
                        difflib.unified_diff(
                            text.splitlines(),
                            new_val.splitlines(),
                            fromfile=f"a/{path_val}",
                            tofile=f"b/{path_val}",
                            lineterm="",
                        )
                    )
                    results.append(
                        f"[{i}] {path_val}: dry run\n{diff or '(no changes)'}"
                    )
                else:
                    p.write_text(new_val, encoding="utf-8")
                    results.append(f"[{i}] {path_val}: overwrote")
            else:
                old_s = old_str if isinstance(old_str, str) else str(old_str)
                new_s = new_text if isinstance(new_text, str) else str(new_text)
                if old_s not in text:
                    results.append(f"[{i}] {path_val}: old_string not found")
                    continue
                new_content = text.replace(old_s, new_s, 1)
                if os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1":
                    diff = "\n".join(
                        difflib.unified_diff(
                            text.splitlines(),
                            new_content.splitlines(),
                            fromfile=f"a/{path_val}",
                            tofile=f"b/{path_val}",
                            lineterm="",
                        )
                    )
                    results.append(
                        f"[{i}] {path_val}: dry run\n{diff or '(no changes)'}"
                    )
                else:
                    p.write_text(new_content, encoding="utf-8")
                    results.append(f"[{i}] {path_val}: replaced")
        except Exception as e:
            results.append(f"[{i}] {path_val}: {e}")
    return "\n".join(results)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
