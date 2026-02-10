"""
Skills and memory: Cursor-style persistent context.

- Skills = Markdown files in ~/.ollamacode/skills (global) and .ollamacode/skills (workspace).
- Loaded into system prompt at start; model can read/write via MCP tools.
- save_memory appends to a single "memory" store (skill or file) for facts/preferences.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _global_skills_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".ollamacode" / "skills"


def _workspace_skills_dir(
    workspace_root: str | None, must_exist: bool = True
) -> Path | None:
    if not workspace_root:
        return None
    p = Path(workspace_root).resolve() / ".ollamacode" / "skills"
    return p if (not must_exist or p.exists()) else None


def get_skills_dirs(workspace_root: str | None = None) -> list[Path]:
    """Return ordered list of skills directories: global first, then workspace if present."""
    out: list[Path] = []
    global_dir = _global_skills_dir()
    if global_dir.exists():
        out.append(global_dir)
    ws = _workspace_skills_dir(workspace_root)
    if ws is not None:
        out.append(ws)
    return out


def _safe_skill_name(name: str) -> str:
    """Allow only alphanumeric, underscore, hyphen; no path traversal."""
    name = (name or "").strip()
    if not name or name in (".", ".."):
        return ""
    if re.search(r"[^\w\-]", name):
        return ""
    return name


def _skill_path(skills_dir: Path, name: str) -> Path | None:
    safe = _safe_skill_name(name)
    if not safe:
        return None
    return skills_dir / f"{safe}.md"


def list_skills(workspace_root: str | None = None) -> list[str]:
    """List skill names (without .md) from global and workspace skills dirs, deduplicated (workspace overrides same name)."""
    seen: set[str] = set()
    names: list[str] = []
    for d in get_skills_dirs(workspace_root):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == ".md" and f.is_file():
                name = f.stem
                if _safe_skill_name(name) and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def read_skill(name: str, workspace_root: str | None = None) -> str | None:
    """Read skill content. Check workspace first, then global. Returns None if not found or invalid name."""
    safe = _safe_skill_name(name)
    if not safe:
        return None
    dirs = get_skills_dirs(workspace_root)
    for d in reversed(dirs):
        path = d / f"{safe}.md"
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
    return None


def _write_dir(workspace_root: str | None) -> Path:
    """Directory to write new skills: workspace .ollamacode/skills if workspace_root set, else global."""
    ws = _workspace_skills_dir(workspace_root, must_exist=False)
    if ws is not None:
        return ws
    global_dir = _global_skills_dir()
    global_dir.mkdir(parents=True, exist_ok=True)
    return global_dir


def write_skill(name: str, content: str, workspace_root: str | None = None) -> str:
    """Write or overwrite a skill file. Returns a short status message."""
    safe = _safe_skill_name(name)
    if not safe:
        return (
            "Error: invalid skill name (use letters, numbers, underscore, hyphen only)."
        )
    d = _write_dir(workspace_root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{safe}.md"
    try:
        path.write_text((content or "").strip() + "\n", encoding="utf-8")
        return f"Wrote skill '{safe}' ({len((content or '').strip())} chars) to {path}"
    except OSError as e:
        return f"Error writing skill: {e}"


MEMORY_SKILL_NAME = "memory"


def save_memory(key: str, value: str, workspace_root: str | None = None) -> str:
    """Append a key-value fact to the memory skill (or create it). Use for preferences, decisions, facts."""
    existing = read_skill(MEMORY_SKILL_NAME, workspace_root) or ""
    line = f"- **{key}**: {value}\n"
    if existing.strip():
        new_content = existing.rstrip() + "\n" + line
    else:
        new_content = "# Memory\n\n" + line
    return write_skill(MEMORY_SKILL_NAME, new_content, workspace_root)


def load_skills_text(workspace_root: str | None = None) -> str:
    """Load all skills as one block for system prompt. Each skill as a section with its name."""
    parts: list[str] = []
    for d in get_skills_dirs(workspace_root):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == ".md" and f.is_file():
                name = f.stem
                if not _safe_skill_name(name):
                    continue
                try:
                    text = f.read_text(encoding="utf-8").strip()
                    if text:
                        parts.append(f"--- Skill: {name} ---\n\n{text}")
                except OSError:
                    continue
    if not parts:
        return ""
    return "\n\n".join(parts)
