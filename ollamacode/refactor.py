"""Refactor helpers: rename symbols and generate diffs."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Iterable

from .symbol_index import find_references, query_symbol
from .edits import reverse_unified_diff, apply_unified_diff_filtered


def _ts_python_parser():
    """Return a tree-sitter Parser for Python if available."""
    try:
        from tree_sitter_languages import get_language  # type: ignore[import-not-found]
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        language = get_language("python")
        parser = Parser()
        parser.set_language(language)
        return parser
    except Exception:
        return None


def _ts_end_line(node) -> int:
    end_row, end_col = node.end_point
    # tree-sitter end_point is exclusive
    return end_row if end_col == 0 else end_row + 1


def _ts_is_statement(node) -> bool:
    if not node.is_named:
        return False
    if node.type in ("function_definition", "class_definition", "decorated_definition"):
        return True
    return node.type.endswith("_statement")


def _ts_extract_statement_range(
    text: str, start_line: int, end_line: int
) -> tuple[int, int] | None:
    """Use tree-sitter to snap a selection to full statement boundaries."""
    parser = _ts_python_parser()
    if parser is None:
        return None
    tree = parser.parse(bytes(text, "utf-8"))
    root = tree.root_node
    start_row = start_line - 1
    end_row = end_line - 1
    try:
        node = root.named_descendant_for_point_range(
            (start_row, 0), (end_row, 0)
        )
    except Exception:
        node = root
    parent = node
    while parent is not None:
        if any(_ts_is_statement(c) for c in parent.named_children):
            break
        parent = parent.parent
    if parent is None:
        return None
    stmt_children = [c for c in parent.named_children if _ts_is_statement(c)]
    if not stmt_children:
        return None

    def _within(n) -> bool:
        return n.start_point[0] >= start_row and _ts_end_line(n) - 1 <= end_row

    def _overlaps(n) -> bool:
        return n.start_point[0] <= end_row and _ts_end_line(n) - 1 >= start_row

    selected = [c for c in stmt_children if _within(c)]
    if not selected:
        return None
    for c in stmt_children:
        if _overlaps(c) and c not in selected:
            return None
    idxs = [stmt_children.index(c) for c in selected]
    if max(idxs) - min(idxs) + 1 != len(idxs):
        return None

    new_start = min(c.start_point[0] for c in selected) + 1
    new_end = max(_ts_end_line(c) for c in selected)
    return new_start, new_end


def _iter_text_files(root: Path, max_files: int = 500) -> list[Path]:
    from .repo_map import _iter_files

    return _iter_files(root, max_files=max_files)


def rename_symbol(
    workspace_root: str,
    old_name: str,
    new_name: str,
    *,
    max_files: int = 500,
) -> list[dict]:
    """Rename symbol occurrences. Returns edits list with unified diffs."""
    root = Path(workspace_root).resolve()
    if not old_name or not new_name or old_name == new_name:
        return []

    # Try symbol index first for targeted files.
    files: set[Path] = set()
    try:
        defs = query_symbol(old_name, workspace_root=str(root))
        refs = find_references(old_name, workspace_root=str(root))
        for row in defs + refs:
            p = root / str(row.get("path", ""))
            if p.exists():
                files.add(p.resolve())
    except Exception:
        files = set()

    if not files:
        for p in _iter_text_files(root, max_files=max_files):
            files.add(p.resolve())

    edits: list[dict] = []
    pattern = re.compile(rf"\b{re.escape(old_name)}\b")
    for path in sorted(files):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not pattern.search(text):
            continue
        new_text = pattern.sub(new_name, text)
        if new_text == text:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        diff = "\n".join(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        if diff:
            edits.append({"path": rel, "newText": diff})
    return edits


def extract_function(
    file_path: str,
    start_line: int,
    end_line: int,
    new_name: str,
) -> str | None:
    """Extract a block of lines into a new function (AST-aware when possible)."""
    path = Path(file_path)
    if not path.exists() or start_line <= 0 or end_line < start_line:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if end_line > len(lines):
        return None
    snapped = _ts_extract_statement_range(text, start_line, end_line)
    if snapped is not None:
        start_line, end_line = snapped
    block = lines[start_line - 1 : end_line]
    if not block:
        return None
    # Determine indentation from first non-empty line
    indent = ""
    for line in block:
        if line.strip():
            indent = line[: len(line) - len(line.lstrip())]
            break
    body = [indent + "    " + l.lstrip() for l in block if l.strip() or l == ""]
    func_lines = [indent + f"def {new_name}():", *body, ""]
    # Replace block with function + call (define before call)
    call_line = indent + f"{new_name}()"
    new_lines = (
        lines[: start_line - 1]
        + func_lines
        + [call_line]
        + lines[end_line:]
    )
    try:
        path.write_text("\n".join(new_lines), encoding="utf-8")
    except OSError:
        return None
    return str(path)


def move_symbol(
    source_file: str,
    symbol_name: str,
    target_file: str,
) -> bool:
    """Move a top-level function/class to another file (tree-sitter if available; regex fallback)."""
    src = Path(source_file)
    dst = Path(target_file)
    if not src.exists() or not symbol_name:
        return False
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    func_text = None
    new_text = text
    # tree-sitter path
    parser = _ts_python_parser()
    if parser is not None:
        try:
            tree = parser.parse(bytes(text, "utf-8"))
            root = tree.root_node
            for node in root.children:
                target_node = None
                if node.type in ("function_definition", "class_definition"):
                    target_node = node
                elif node.type == "decorated_definition":
                    target_node = next(
                        (c for c in node.named_children if c.type in ("function_definition", "class_definition")),
                        None,
                    )
                if target_node is None:
                    continue
                name_node = next(
                    (c for c in target_node.children if c.type == "identifier"), None
                )
                if name_node and name_node.text.decode("utf-8") == symbol_name:
                    start, end = node.start_byte, node.end_byte
                    func_text = text[start:end]
                    new_text = text[:start] + text[end:]
                    break
        except Exception:
            pass
    # fallback: regex block by indentation
    if func_text is None:
        lines = text.splitlines()
        start_idx = None
        indent = ""
        for i, line in enumerate(lines):
            m = re.match(
                rf"^(\s*)(def|class)\s+{re.escape(symbol_name)}\s*(\(|:)",
                line,
            )
            if m:
                start_idx = i
                indent = m.group(1)
                break
        if start_idx is None:
            return False
        end_idx = start_idx + 1
        while end_idx < len(lines):
            if lines[end_idx].startswith(indent) and lines[end_idx].strip() and not lines[end_idx].startswith(indent + " "):
                break
            end_idx += 1
        func_lines = lines[start_idx:end_idx]
        func_text = "\n".join(func_lines)
        new_lines = lines[:start_idx] + lines[end_idx:]
        new_text = "\n".join(new_lines)
    if func_text is None:
        return False
    try:
        src.write_text(new_text.strip() + "\n", encoding="utf-8")
    except OSError:
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        existing = dst.read_text(encoding="utf-8", errors="replace") if dst.exists() else ""
        sep = "\n\n" if existing and not existing.endswith("\n") else "\n"
        dst.write_text(existing + sep + func_text.strip() + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def move_function(
    source_file: str,
    function_name: str,
    target_file: str,
) -> bool:
    """Backward-compatible wrapper for move_symbol (functions/classes)."""
    return move_symbol(source_file, function_name, target_file)

def save_last_refactor(diff_text: str) -> str | None:
    """Persist last refactor diff for rollback."""
    try:
        path = Path.home() / ".ollamacode" / "last_refactor.diff"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(diff_text, encoding="utf-8")
        return str(path)
    except OSError:
        return None


def rollback_last_refactor(workspace_root: str) -> int:
    """Rollback last refactor by applying reverse diff."""
    path = Path.home() / ".ollamacode" / "last_refactor.diff"
    if not path.exists():
        return 0
    try:
        diff_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    reverse = reverse_unified_diff(diff_text)
    return apply_unified_diff_filtered(reverse, Path(workspace_root).resolve(), include_hunk=None)
