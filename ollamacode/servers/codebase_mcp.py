"""
Built-in codebase search MCP server: search_codebase, get_relevant_files, glob, grep.

Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP("ollamacode-codebase")

# Limits to keep responses small
MAX_RESULTS = 50
MAX_FILE_BYTES = 500_000
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
_REGEX_MAX_LENGTH = 500  # Max regex pattern length to prevent ReDoS


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _resolve(path: str) -> Path:
    p = _root() / path.lstrip("/")
    resolved = p.resolve()
    if not resolved.is_relative_to(_root()):
        raise ValueError(f"Path {path!r} is outside workspace root {_root()}")
    return resolved


def _matches_file_types(path: Path, file_types: list[str] | None) -> bool:
    """Check if a file path matches the given file type extensions."""
    if not file_types:
        return True
    suffix = path.suffix.lower()
    return suffix in {ft if ft.startswith(".") else f".{ft}" for ft in file_types}


@mcp.tool()
def search_codebase(
    query: str,
    file_pattern: str = "*",
    max_results: int = MAX_RESULTS,
    file_types: list[str] | None = None,
) -> str:
    """
    Search the workspace for a text query (case-insensitive substring match).
    Returns matching file paths with line numbers and snippet context.
    file_pattern: glob for files to search (default '*' = all). Examples: '*.py', '*.ts'.
    max_results: cap on number of matches returned (default 50).
    file_types: optional list of file extensions to filter by (e.g. [".py", ".js"]). When set, only files with matching extensions are searched.
    """
    max_results = max(1, min(max_results, MAX_RESULTS))
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
            if not _matches_file_types(path, file_types):
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
def search_incremental(
    query: str,
    offset: int = 0,
    limit: int = 20,
    file_pattern: str = "*",
    file_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Incremental (paginated) search: like search_codebase but returns a page of results.
    query: text to search for (case-insensitive).
    offset: number of matches to skip (default 0).
    limit: max results to return in this page (default 20, max 50).
    file_pattern: glob for files to search (default '*').
    file_types: optional list of file extensions to filter by (e.g. [".py", ".js"]).
    Returns: {results: [{path, line, snippet}], total: int, offset: int, has_more: bool}
    """
    limit = max(1, min(limit, MAX_RESULTS))
    offset = max(0, offset)
    root = _root()
    query_lower = query.lower()
    # Collect all matches up to offset + limit + 1 to know if there are more
    all_matches: list[tuple[str, int, str]] = []
    need = offset + limit + 1

    try:
        pattern = file_pattern.strip() or "*"
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if not _matches_file_types(path, file_types):
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
                    all_matches.append((rel_str, i, line.strip()[:200]))
                    if len(all_matches) >= need:
                        break
            if len(all_matches) >= need:
                break
    except Exception as e:
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "has_more": False,
            "error": str(e),
        }

    total_found = len(all_matches)
    page = all_matches[offset : offset + limit]
    has_more = total_found > offset + limit

    return {
        "results": [{"path": p, "line": n, "snippet": s} for p, n, s in page],
        "total": total_found
        if not has_more
        else total_found - 1,  # total is at least this many
        "offset": offset,
        "has_more": has_more,
    }


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


@mcp.tool()
def glob(pattern: str, limit: int = 200) -> str:
    """
    List files in the workspace matching a glob pattern.
    pattern: e.g. '*.py', '**/*.md', 'src/**/*.ts'. Paths relative to workspace root.
    limit: max number of paths to return (default 200).
    """
    root = _root()
    if not pattern.strip():
        return "Provide a glob pattern (e.g. *.py, **/*.md)."
    pat = pattern.strip().lstrip("/")
    if pat.startswith("**/"):
        pat = pat[3:]
    results: list[str] = []
    try:
        for path in root.rglob(pat):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            results.append(str(path.relative_to(root)))
            if len(results) >= limit:
                break
    except Exception as e:
        return f"Glob error: {e}"
    if not results:
        return f"No files matching {pattern!r}."
    return "\n".join(sorted(results))


@mcp.tool()
def grep(
    pattern: str,
    path: str = ".",
    context_lines: int = 0,
    max_results: int = 100,
    file_types: list[str] | None = None,
) -> str:
    """
    Regex search in workspace files. Returns matching lines with optional context.
    pattern: regex (Python re). path: directory or file to search (default '.'). context_lines: lines before/after each match (default 0). max_results: cap matches (default 100).
    file_types: optional list of file extensions to filter by (e.g. [".py", ".js"]). When set, only files with matching extensions are searched.
    """
    root = _root()
    if len(pattern) > _REGEX_MAX_LENGTH:
        return f"Regex pattern too long ({len(pattern)} chars, max {_REGEX_MAX_LENGTH})"
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    target = _resolve(path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = [
            p
            for p in target.rglob("*")
            if p.is_file()
            and not any(s in p.parts for s in SKIP_DIRS)
            and _matches_file_types(p, file_types)
        ]
    else:
        return f"Path not found: {path}"
    ctx = max(0, min(context_lines, 5))
    results: list[str] = []
    try:
        for fp in files[:500]:
            if len(results) >= max_results:
                break
            try:
                if fp.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = fp.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            rel = str(fp.relative_to(root))
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if len(results) >= max_results:
                    break
                if compiled.search(line):
                    start = max(0, i - ctx)
                    end = min(len(lines), i + ctx + 1)
                    block = "\n".join(
                        f"  {start + j + 1}: {lines[j]}" for j in range(start, end)
                    )
                    results.append(f"{rel}:{i + 1}:\n{block}")
    except Exception as e:
        return f"Grep error: {e}"
    if not results:
        return f"No matches for {pattern!r}."
    return "\n---\n".join(results)


@mcp.tool()
def build_repo_map(
    max_files: int = 200,
    max_symbols_per_file: int = 6,
    max_chars_per_file: int = 6000,
) -> str:
    """
    Build a compact repo map with top-level symbols.
    Returns markdown text (does not write files).
    """
    try:
        from ollamacode.repo_map import build_repo_map as _build

        return _build(
            str(_root()),
            max_files=max(1, min(max_files, 2000)),
            max_symbols_per_file=max(1, min(max_symbols_per_file, 20)),
            max_chars_per_file=max(1000, min(max_chars_per_file, 20000)),
        )
    except Exception as e:
        return f"Repo map error: {e}"


@mcp.tool()
def build_symbol_index(
    max_files: int = 400,
    max_symbols_per_file: int = 20,
    max_chars_per_file: int = 12000,
) -> dict[str, list[str]]:
    """Build a symbol index {path: [symbols...]}. Useful for quick lookup."""
    try:
        from ollamacode.repo_map import build_symbol_index as _build

        return _build(
            str(_root()),
            max_files=max(1, min(max_files, 5000)),
            max_symbols_per_file=max(1, min(max_symbols_per_file, 50)),
            max_chars_per_file=max(1000, min(max_chars_per_file, 30000)),
        )
    except Exception as e:
        return {"error": [str(e)]}


@mcp.tool()
def build_symbol_graph(
    max_files: int = 400,
    max_chars_per_file: int = 12000,
) -> dict[str, Any]:
    """Build a symbol graph with definitions and call references (best-effort)."""
    try:
        from ollamacode.symbol_graph import build_symbol_graph as _build

        return _build(
            str(_root()),
            max_files=max(1, min(max_files, 5000)),
            max_chars_per_file=max(1000, min(max_chars_per_file, 30000)),
        )
    except Exception as e:
        return {"error": {"message": str(e)}}


@mcp.tool()
def index_symbols(
    max_files: int = 400,
    max_chars_per_file: int = 12000,
) -> dict[str, Any]:
    """Build persistent symbol index (definitions + references)."""
    try:
        from ollamacode.symbol_index import build_symbol_index as _build

        return _build(
            str(_root()),
            max_files=max(1, min(max_files, 5000)),
            max_chars_per_file=max(1000, min(max_chars_per_file, 30000)),
        )
    except Exception as e:
        return {"symbols": 0, "references": 0, "error": str(e)}


@mcp.tool()
def query_symbol_index(name: str, limit: int = 50) -> dict[str, Any]:
    """Query persistent symbol index for definitions."""
    try:
        from ollamacode.symbol_index import query_symbol as _query

        rows = _query(name, workspace_root=str(_root()), limit=max(1, min(limit, 200)))
        return {"matches": rows}
    except Exception as e:
        return {"matches": [], "error": str(e)}


@mcp.tool()
def find_symbol_references(name: str, limit: int = 100) -> dict[str, Any]:
    """Find references to a symbol using persistent index."""
    try:
        from ollamacode.symbol_index import find_references as _find

        rows = _find(name, workspace_root=str(_root()), limit=max(1, min(limit, 500)))
        return {"matches": rows}
    except Exception as e:
        return {"matches": [], "error": str(e)}


@mcp.tool()
def refactor_rename(
    old_name: str, new_name: str, max_files: int = 500
) -> dict[str, Any]:
    """Generate unified-diff edits to rename a symbol across the workspace."""
    try:
        from ollamacode.refactor import rename_symbol

        edits = rename_symbol(str(_root()), old_name, new_name, max_files=max_files)
        return {"edits": edits}
    except Exception as e:
        return {"edits": [], "error": str(e)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
