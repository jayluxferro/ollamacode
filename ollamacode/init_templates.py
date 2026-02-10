# init_templates.py
# Contains minimal template definitions used by the OllamaCode project.
# The dictionary maps relative file paths to their content.
# For the purposes of this repository, an empty mapping is sufficient.

from __future__ import annotations

from pathlib import Path

TEMPLATES: dict[str, dict[str, str]] = {}
TEMPLATE_NAMES = ["python-cli", "python-lib", "web-app", "rust-cli", "go-mod"]


def run_init(template: str | None, dest: str) -> str:
    """List templates or scaffold a project. Returns a message string."""
    if template is None:
        return "Available templates: " + ", ".join(TEMPLATE_NAMES) + "\nUsage: ollamacode init --template <name> [--dest DIR]"
    if template not in TEMPLATE_NAMES:
        return f"Unknown template: {template!r}. Available: " + ", ".join(TEMPLATE_NAMES)
    files = TEMPLATES.get(template, {})
    if not files:
        return f"Template {template!r} has no files defined yet."
    root = Path(dest).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return f"Created {len(files)} file(s) in {root}."
