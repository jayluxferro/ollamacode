"""Repository map utilities: compact overview of files and top-level symbols."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".cursor",
    "dist",
    "build",
}

_TEXT_EXTS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}


def _iter_files(root: Path, max_files: int) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if len(out) >= max_files:
            break
        if not p.is_file():
            continue
        if any(part in _IGNORE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in _TEXT_EXTS:
            continue
        out.append(p)
    return out


def _extract_symbols_regex(text: str, max_symbols: int = 6) -> list[str]:
    patterns: Iterable[tuple[str, int]] = [
        (r"^\s*def\s+([A-Za-z_][\w]*)\s*\(", 1),
        (r"^\s*class\s+([A-Za-z_][\w]*)\s*[:\(]", 1),
        (r"^\s*function\s+([A-Za-z_][\w]*)\s*\(", 1),
        (r"^\s*export\s+function\s+([A-Za-z_][\w]*)\s*\(", 1),
        (r"^\s*const\s+([A-Za-z_][\w]*)\s*=\s*\(", 1),
        (r"^\s*func\s+([A-Za-z_][\w]*)\s*\(", 1),
        (r"^\s*fn\s+([A-Za-z_][\w]*)\s*\(", 1),
        (r"^\s*struct\s+([A-Za-z_][\w]*)\s*[{<]", 1),
        (r"^\s*enum\s+([A-Za-z_][\w]*)\s*[{<]", 1),
    ]
    symbols: list[str] = []
    for line in text.splitlines():
        for pat, group in patterns:
            m = re.match(pat, line)
            if m:
                symbols.append(m.group(group))
                if len(symbols) >= max_symbols:
                    return symbols
    return symbols


def _extract_symbols_for_file(text: str, suffix: str, max_symbols: int = 6) -> list[str]:
    """Extract symbols using tree-sitter if available, else regex fallback."""
    try:
        from tree_sitter_languages import get_language  # type: ignore[import-not-found]
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except Exception:
        return _extract_symbols_regex(text, max_symbols=max_symbols)

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
        return _extract_symbols_regex(text, max_symbols=max_symbols)
    try:
        language = get_language(lang_name)
        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(text, "utf-8"))
        root = tree.root_node
    except Exception:
        return _extract_symbols_regex(text, max_symbols=max_symbols)

    symbols: list[str] = []

    def visit(node):
        nonlocal symbols
        if len(symbols) >= max_symbols:
            return
        if node.type in ("function_definition", "class_definition"):
            for child in node.children:
                if child.type == "identifier":
                    symbols.append(child.text.decode("utf-8"))
                    return
        if node.type in ("function_declaration", "class_declaration", "method_definition"):
            for child in node.children:
                if child.type in ("identifier", "property_identifier"):
                    symbols.append(child.text.decode("utf-8"))
                    return
        if node.type in ("function_item", "struct_item", "enum_item"):
            for child in node.children:
                if child.type == "identifier":
                    symbols.append(child.text.decode("utf-8"))
                    return
        for c in node.children:
            visit(c)

    visit(root)
    return symbols if symbols else _extract_symbols_regex(text, max_symbols=max_symbols)


def build_symbol_index(
    workspace_root: str,
    *,
    max_files: int = 400,
    max_symbols_per_file: int = 20,
    max_chars_per_file: int = 12000,
) -> dict[str, list[str]]:
    """Return a symbol index: {relative_path: [symbols...]}."""
    root = Path(workspace_root).resolve()
    files = _iter_files(root, max_files=max_files)
    out: dict[str, list[str]] = {}
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
        except OSError:
            continue
        symbols = _extract_symbols_for_file(
            text, path.suffix, max_symbols=max_symbols_per_file
        )
        if symbols:
            out[rel] = symbols
    return out


def build_repo_map(
    workspace_root: str,
    *,
    max_files: int = 200,
    max_symbols_per_file: int = 6,
    max_chars_per_file: int = 6000,
) -> str:
    """Return a compact markdown repo map."""
    root = Path(workspace_root).resolve()
    files = _iter_files(root, max_files=max_files)
    lines: list[str] = ["# Repo Map", ""]
    lines.append(f"- Root: {root}")
    lines.append(f"- Files indexed: {len(files)}")
    lines.append("")
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
        except OSError:
            continue
        symbols = _extract_symbols_for_file(
            text, path.suffix, max_symbols=max_symbols_per_file
        )
        sym_part = f" (symbols: {', '.join(symbols)})" if symbols else ""
        lines.append(f"- {rel}{sym_part}")
    return "\n".join(lines)


def write_repo_map(
    workspace_root: str,
    output_path: str,
    *,
    max_files: int = 200,
    max_symbols_per_file: int = 6,
    max_chars_per_file: int = 6000,
) -> str:
    """Build and write repo map. Returns output path."""
    content = build_repo_map(
        workspace_root,
        max_files=max_files,
        max_symbols_per_file=max_symbols_per_file,
        max_chars_per_file=max_chars_per_file,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)
