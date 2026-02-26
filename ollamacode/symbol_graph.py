"""Symbol graph utilities: definitions and call references (best-effort)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .repo_map import _iter_files


def _regex_defs(text: str) -> list[str]:
    patterns: Iterable[str] = [
        r"^\s*def\s+([A-Za-z_][\w]*)\s*\(",
        r"^\s*class\s+([A-Za-z_][\w]*)\s*[:\(]",
        r"^\s*function\s+([A-Za-z_][\w]*)\s*\(",
        r"^\s*export\s+function\s+([A-Za-z_][\w]*)\s*\(",
    ]
    out: list[str] = []
    for line in text.splitlines():
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                out.append(m.group(1))
    return out


def _regex_calls(text: str) -> list[str]:
    # naive "name(" matches, excluding def/class lines
    calls: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*(def|class|function|export\s+function)\s+", line):
            continue
        for m in re.finditer(r"\b([A-Za-z_][\w]*)\s*\(", line):
            calls.append(m.group(1))
    return calls


def _tree_sitter_calls(text: str, suffix: str) -> list[str] | None:
    try:
        from tree_sitter_languages import get_language  # type: ignore[import-not-found]
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except Exception:
        return None
    lang_name_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }
    lang_name = lang_name_map.get(suffix.lower())
    if not lang_name:
        return None
    try:
        language = get_language(lang_name)
        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(text, "utf-8"))
    except Exception:
        return None
    calls: list[str] = []

    def visit(node):
        if node.type in ("call", "call_expression", "function_call", "call_expression"):
            for child in node.children:
                if child.type in ("identifier", "property_identifier", "field_identifier"):
                    calls.append(child.text.decode("utf-8"))
                    break
        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return calls


def build_symbol_graph(
    workspace_root: str,
    *,
    max_files: int = 400,
    max_chars_per_file: int = 12000,
) -> dict[str, dict[str, list[str]]]:
    """Return {definitions, callers, calls_by_file} graph using regex fallback."""
    root = Path(workspace_root).resolve()
    files = _iter_files(root, max_files=max_files)
    definitions: dict[str, list[str]] = {}
    callers: dict[str, list[str]] = {}
    calls_by_file: dict[str, list[str]] = {}
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
        except OSError:
            continue
        defs = _regex_defs(text)
        calls = _tree_sitter_calls(text, path.suffix) or _regex_calls(text)
        if defs:
            for d in defs:
                definitions.setdefault(d, []).append(rel)
        if calls:
            calls_by_file[rel] = calls
            for c in calls:
                callers.setdefault(c, []).append(rel)
    return {
        "definitions": definitions,
        "callers": callers,
        "calls_by_file": calls_by_file,
    }
