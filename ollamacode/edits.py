"""
Structured apply-edits: parse <<EDITS>> JSON from model output and apply/reject.

Format: <<EDITS>>
[{ "path": "file.py", "oldText": "optional", "newText": "new content" }]
<<END>>

If oldText is omitted, newText is the full file content. If oldText is present, replace
first occurrence with newText.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any


EDITS_START = "<<EDITS>>"
EDITS_END = "<<END>>"


def parse_edits(response_text: str) -> list[dict[str, Any]]:
    """
    Extract <<EDITS>> ... <<END>> block from response and parse as JSON array.
    Each item: path (str), newText (str), oldText (str, optional).
    Returns [] if no block or invalid JSON.
    """
    start = response_text.find(EDITS_START)
    if start == -1:
        return []
    after_start = response_text.index("\n", start) if "\n" in response_text[start:] else start + len(EDITS_START)
    end_marker = response_text.find(EDITS_END, after_start)
    if end_marker == -1:
        return []
    raw = response_text[after_start:end_marker].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        new_text = item.get("newText")
        if path is None or new_text is None:
            continue
        out.append({
            "path": str(path),
            "oldText": item.get("oldText"),
            "newText": new_text if isinstance(new_text, str) else str(new_text),
        })
    return out


def format_edits_diff(edits: list[dict[str, Any]], workspace_root: str | Path) -> str:
    """Produce a human-readable diff summary for the given edits."""
    root = Path(workspace_root).resolve()
    lines: list[str] = []
    for e in edits:
        path = e["path"]
        new_text = e["newText"]
        old_text = e.get("oldText")
        resolved = (root / path).resolve()
        try:
            if not resolved.is_relative_to(root) and resolved != root:
                lines.append(f"  {path}: (outside workspace, skipped)")
                continue
        except (ValueError, TypeError):
            lines.append(f"  {path}: (invalid path)")
            continue
        if old_text is None or old_text == "":
            lines.append(f"  {path}: full file replace ({len(new_text)} chars)")
            continue
        try:
            current = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            lines.append(f"  {path}: (cannot read)")
            continue
        if old_text not in current:
            lines.append(f"  {path}: oldText not found (no change)")
            continue
        diff = "\n".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                current.replace(old_text, new_text, 1).splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        if diff:
            lines.append(f"  {path}:")
            for line in diff.split("\n")[:30]:
                lines.append("    " + line)
            if len(diff.split("\n")) > 30:
                lines.append("    ...")
        else:
            lines.append(f"  {path}: (no diff)")
    return "Edits:\n" + "\n".join(lines) if lines else "No edits to show."


def apply_edits(edits: list[dict[str, Any]], workspace_root: str | Path) -> int:
    """
    Apply edits under workspace_root. Returns number of files written.
    Paths are resolved relative to workspace_root. Skips paths outside workspace.
    """
    root = Path(workspace_root).resolve()
    applied = 0
    for e in edits:
        path = e["path"]
        new_text = e["newText"]
        old_text = e.get("oldText")
        resolved = (root / path).resolve()
        try:
            if not resolved.is_relative_to(root) and resolved != root:
                continue
        except (ValueError, TypeError):
            continue
        try:
            if old_text is None or old_text == "":
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(new_text, encoding="utf-8")
                applied += 1
                continue
            current = resolved.read_text(encoding="utf-8", errors="replace")
            if old_text not in current:
                continue
            new_content = current.replace(old_text, new_text, 1)
            resolved.write_text(new_content, encoding="utf-8")
            applied += 1
        except OSError:
            continue
    return applied
