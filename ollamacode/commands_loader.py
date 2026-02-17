"""
Load custom slash commands from ~/.ollamacode/commands.md and .ollamacode/commands.md.

Format: ## /name
Short description.

Optional prompt template (use {{rest}} for the user's text after the command).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class CustomCommand(NamedTuple):
    name: str  # e.g. "/mycmd"
    description: str
    prompt_template: str | None  # None = no template, just show description


def load_custom_commands(workspace_root: str | Path) -> list[CustomCommand]:
    """
    Load custom slash commands from user and project commands.md.
    User: ~/.ollamacode/commands.md. Project: .ollamacode/commands.md or workspace_root/commands.md.
    Returns list of CustomCommand (name, description, prompt_template).
    """
    root = Path(workspace_root).resolve()
    candidates = [
        Path.home() / ".ollamacode" / "commands.md",
        root / ".ollamacode" / "commands.md",
        root / "commands.md",
    ]
    seen_names: set[str] = set()
    result: list[CustomCommand] = []
    for path in candidates:
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for cmd in _parse_commands_md(text):
                if cmd.name.lower() not in seen_names:
                    seen_names.add(cmd.name.lower())
                    result.append(cmd)
        except OSError:
            continue
    return result


def _parse_commands_md(text: str) -> list[CustomCommand]:
    """Parse markdown with ## /name blocks. Returns list of CustomCommand."""
    out: list[CustomCommand] = []
    blocks = text.split("## ")
    for block in blocks:
        block = block.strip()
        if not block or not block.startswith("/"):
            continue
        lines = block.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if not first.startswith("/"):
            continue
        name = first.split()[0].strip() if first else ""
        if not name.startswith("/"):
            continue
        desc_lines: list[str] = []
        template_lines: list[str] = []
        in_template = False
        for line in lines[1:]:
            if line.strip() == "" and not in_template:
                in_template = True
                continue
            if in_template:
                template_lines.append(line)
            else:
                desc_lines.append(line)
        description = (
            " ".join(ln.strip() for ln in desc_lines if ln.strip()).strip() or name
        )
        prompt_template = "\n".join(template_lines).strip() or None
        out.append(
            CustomCommand(
                name=name, description=description, prompt_template=prompt_template
            )
        )
    return out
