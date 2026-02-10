"""
Prompt templates: load Markdown from ~/.ollamacode/templates or .ollamacode/templates.

Config: prompt_template: "refactor" loads refactor.md. Used to append task-specific system instructions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _global_templates_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".ollamacode" / "templates"


def _template_path(name: str, workspace_root: str | None) -> Path | None:
    """Resolve template name to file path. Workspace .ollamacode/templates first, then global. Name must be safe (alphanumeric, -, _)."""
    safe = (name or "").strip()
    if not safe or re.search(r"[^\w\-]", safe):
        return None
    if workspace_root:
        ws = Path(workspace_root).resolve() / ".ollamacode" / "templates" / f"{safe}.md"
        if ws.is_file():
            return ws
    global_dir = _global_templates_dir()
    p = global_dir / f"{safe}.md"
    return p if p.is_file() else None


def load_prompt_template(name: str, workspace_root: str | None = None) -> str:
    """Load template content by name. Returns empty string if not found or invalid name."""
    path = _template_path(name, workspace_root)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def list_templates(workspace_root: str | None = None) -> list[str]:
    """List available template names (without .md) from global and workspace."""
    seen: set[str] = set()
    names: list[str] = []
    for base in (_global_templates_dir(),):
        if base.is_dir():
            for f in sorted(base.iterdir()):
                if f.suffix == ".md" and f.is_file() and re.match(r"^[\w\-]+$", f.stem):
                    if f.stem not in seen:
                        seen.add(f.stem)
                        names.append(f.stem)
    if workspace_root:
        ws = Path(workspace_root).resolve() / ".ollamacode" / "templates"
        if ws.is_dir():
            for f in sorted(ws.iterdir()):
                if f.suffix == ".md" and f.is_file() and re.match(r"^[\w\-]+$", f.stem):
                    if f.stem not in seen:
                        seen.add(f.stem)
                        names.append(f.stem)
    return names
