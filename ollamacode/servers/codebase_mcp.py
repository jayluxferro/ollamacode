"""
Built-in codebase search MCP server: search_codebase, get_relevant_files.

Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-codebase")

# Limits to keep responses small
MAX_RESULTS = 50
MAX_FILE_BYTES = 500_000
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


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
def search_codebase(
    query: str, file_pattern: str = "*", max_results: int = MAX_RESULTS
) -> str:
    """
    Search the workspace for a text query (case-insensitive substring match).
    Returns matching file paths with line numbers and snippet context.
    file_pattern: glob for files to search (default '*' = all). Examples: '*.py', '*.ts'.
    max_results: cap on number of matches returned (default 50).
    """
    root = _root()
    query_lower = query.lower()
    results: list[tuple[str, int, str]] = []

    try:
        pattern = file_pattern.strip() or "*"
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            rel_str = str(path.relative_to(root))
            for i, line in enumerate(text.splitlines(), 1):
                if query_lower in line.lower():
                    results.append((rel_str, i, line.strip()[:200]))
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return f"No matches for {query!r} in workspace."
    lines = [f"{path}:{num}: {snippet}" for path, num, snippet in results]
    return "\n".join(lines)


@mcp.tool()
def get_relevant_files(description: str, limit: int = 20) -> str:
    """
    List files in the workspace that might be relevant to a description (keyword match).
    description: short phrase (e.g. 'auth login', 'config yaml'). Matches against file paths and names.
    limit: max number of file paths to return (default 20).
    """
    root = _root()
    words = [w.lower() for w in description.split() if len(w) > 1]
    if not words:
        return "Provide a short description (e.g. 'auth config')."

    matches: list[str] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            rel = str(path.relative_to(root)).lower()
            if all(w in rel for w in words):
                matches.append(str(path.relative_to(root)))
                if len(matches) >= limit:
                    break
    except Exception as e:
        return f"Error: {e}"

    if not matches:
        return f"No files matching {description!r}."
    return "\n".join(matches)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
