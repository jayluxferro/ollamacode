"""Persistent symbol index with definitions and references."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

from .repo_map import _iter_files

_DB_PATH = Path.home() / ".ollamacode" / "symbols.db"


def _db_path() -> Path:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            workspace TEXT NOT NULL,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refs (
            workspace TEXT NOT NULL,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_symbols_ws ON symbols(workspace);
        CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(name);
        CREATE INDEX IF NOT EXISTS idx_refs_ws ON refs(workspace);
        """
    )
    conn.commit()


def _regex_defs_calls(text: str) -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]]:
    patterns: Iterable[tuple[str, str]] = [
        (r"^\s*def\s+([A-Za-z_][\w]*)\s*\(", "function"),
        (r"^\s*class\s+([A-Za-z_][\w]*)\s*[:\(]", "class"),
        (r"^\s*function\s+([A-Za-z_][\w]*)\s*\(", "function"),
        (r"^\s*export\s+function\s+([A-Za-z_][\w]*)\s*\(", "function"),
    ]
    defs: list[tuple[str, str, int]] = []
    refs: list[tuple[str, int]] = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat, kind in patterns:
            m = re.match(pat, line)
            if m:
                defs.append((m.group(1), kind, i))
        if re.match(r"^\s*(def|class|function|export\s+function)\s+", line):
            continue
        for m in re.finditer(r"\b([A-Za-z_][\w]*)\s*\(", line):
            refs.append((m.group(1), i))
    return defs, refs


def _ts_defs_calls(text: str, suffix: str) -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]] | None:
    try:
        from tree_sitter_languages import get_language  # type: ignore[import-not-found]
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except Exception:
        return None
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }
    lang_name = lang_map.get(suffix.lower())
    if not lang_name:
        return None
    try:
        language = get_language(lang_name)
        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(text, "utf-8"))
    except Exception:
        return None
    defs: list[tuple[str, str, int]] = []
    refs: list[tuple[str, int]] = []

    def visit(node):
        if node.type in ("function_definition", "class_definition"):
            for child in node.children:
                if child.type == "identifier":
                    defs.append((child.text.decode("utf-8"), "function" if node.type == "function_definition" else "class", child.start_point[0] + 1))
                    break
        if node.type in ("function_declaration", "class_declaration", "method_definition"):
            for child in node.children:
                if child.type in ("identifier", "property_identifier"):
                    kind = "function" if "function" in node.type else "class"
                    defs.append((child.text.decode("utf-8"), kind, child.start_point[0] + 1))
                    break
        if node.type in ("call", "call_expression", "function_call"):
            for child in node.children:
                if child.type in ("identifier", "property_identifier", "field_identifier"):
                    refs.append((child.text.decode("utf-8"), child.start_point[0] + 1))
                    break
        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return defs, refs


def build_symbol_index(
    workspace_root: str,
    *,
    max_files: int = 400,
    max_chars_per_file: int = 12000,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Build persistent symbol index. Returns counts."""
    root = Path(workspace_root).resolve()
    files = _iter_files(root, max_files=max_files)
    db = db_path or _db_path()
    conn = sqlite3.connect(str(db))
    try:
        _init_schema(conn)
        conn.execute("DELETE FROM symbols WHERE workspace = ?", (str(root),))
        conn.execute("DELETE FROM refs WHERE workspace = ?", (str(root),))
        sym_count = 0
        ref_count = 0
        for path in files:
            rel = str(path.relative_to(root)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
            except OSError:
                continue
            ts = _ts_defs_calls(text, path.suffix)
            defs, refs = ts if ts is not None else _regex_defs_calls(text)
            for name, kind, line in defs:
                conn.execute(
                    "INSERT INTO symbols (workspace, path, name, kind, line) VALUES (?, ?, ?, ?, ?)",
                    (str(root), rel, name, kind, line),
                )
                sym_count += 1
            for name, line in refs:
                conn.execute(
                    "INSERT INTO refs (workspace, path, name, line) VALUES (?, ?, ?, ?)",
                    (str(root), rel, name, line),
                )
                ref_count += 1
        conn.commit()
        return {"symbols": sym_count, "references": ref_count}
    finally:
        conn.close()


def query_symbol(
    name: str,
    *,
    workspace_root: str,
    db_path: Path | None = None,
    limit: int = 50,
) -> list[dict[str, str | int]]:
    db = db_path or _db_path()
    conn = sqlite3.connect(str(db))
    try:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT path, kind, line FROM symbols WHERE workspace = ? AND name = ? LIMIT ?",
            (str(Path(workspace_root).resolve()), name, limit),
        )
        return [{"path": r[0], "kind": r[1], "line": r[2]} for r in cur.fetchall()]
    finally:
        conn.close()


def find_references(
    name: str,
    *,
    workspace_root: str,
    db_path: Path | None = None,
    limit: int = 100,
) -> list[dict[str, str | int]]:
    db = db_path or _db_path()
    conn = sqlite3.connect(str(db))
    try:
        _init_schema(conn)
        cur = conn.execute(
            "SELECT path, line FROM refs WHERE workspace = ? AND name = ? LIMIT ?",
            (str(Path(workspace_root).resolve()), name, limit),
        )
        return [{"path": r[0], "line": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()
