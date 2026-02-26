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
from typing import Any, Callable

EDITS_START = "<<EDITS>>"
EDITS_END = "<<END>>"

REASONING_START = "<<REASONING>>"
REASONING_END = "<<END>>"

REVIEW_START = "<<REVIEW>>"
REVIEW_END = "<<END>>"


def parse_reasoning(response_text: str) -> tuple[dict[str, Any] | None, str]:
    """
    Extract <<REASONING>> ... <<END>> block; return (reasoning_dict, text_without_block).
    reasoning_dict is { "steps": list[str], "conclusion": str } or None if not present.
    """
    start = response_text.find(REASONING_START)
    if start == -1:
        return (None, response_text)
    after_start = (
        response_text.index("\n", start)
        if "\n" in response_text[start:]
        else start + len(REASONING_START)
    )
    end_marker = response_text.find(REASONING_END, after_start)
    if end_marker == -1:
        return (None, response_text)
    raw = response_text[after_start:end_marker].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return (None, response_text)
    if not isinstance(data, dict):
        return (None, response_text)
    steps = data.get("steps")
    conclusion = data.get("conclusion")
    if not isinstance(steps, list):
        steps = []
    steps = [str(s) for s in steps if s is not None]
    conclusion = str(conclusion).strip() if conclusion is not None else ""
    before = response_text[:start].strip()
    after = response_text[end_marker + len(REASONING_END) :].strip()
    rest = (before + "\n\n" + after).strip() if after else before
    return ({"steps": steps, "conclusion": conclusion}, rest)


def parse_review(response_text: str) -> tuple[list[dict[str, Any]] | None, str]:
    """
    Extract <<REVIEW>> ... <<END>> block; return (suggestions_list, text_without_block).
    suggestions_list is [ {"location": str, "suggestion": str, "rationale": str}, ... ] or None.
    """
    start = response_text.find(REVIEW_START)
    if start == -1:
        return (None, response_text)
    after_start = (
        response_text.index("\n", start)
        if "\n" in response_text[start:]
        else start + len(REVIEW_START)
    )
    end_marker = response_text.find(REVIEW_END, after_start)
    if end_marker == -1:
        return (None, response_text)
    raw = response_text[after_start:end_marker].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return (None, response_text)
    suggestions = data.get("suggestions") if isinstance(data, dict) else None
    if not isinstance(suggestions, list):
        return (None, response_text)
    out: list[dict[str, Any]] = []
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        loc = s.get("location") or s.get("file") or ""
        sug = s.get("suggestion") or s.get("text") or ""
        rat = s.get("rationale") or ""
        out.append(
            {"location": str(loc), "suggestion": str(sug), "rationale": str(rat)}
        )
    before = response_text[:start].strip()
    after = response_text[end_marker + len(REVIEW_END) :].strip()
    rest = (before + "\n\n" + after).strip() if after else before
    return (out, rest)


def parse_edits(response_text: str) -> list[dict[str, Any]]:
    """
    Extract <<EDITS>> ... <<END>> block from response and parse as JSON array.
    Each item: path (str), newText (str), oldText (str, optional).
    Returns [] if no block or invalid JSON.
    """
    start = response_text.find(EDITS_START)
    if start == -1:
        return []
    after_start = (
        response_text.index("\n", start)
        if "\n" in response_text[start:]
        else start + len(EDITS_START)
    )
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
        out.append(
            {
                "path": str(path),
                "oldText": item.get("oldText"),
                "newText": new_text if isinstance(new_text, str) else str(new_text),
            }
        )
    return out


def format_edits_diff(edits: list[dict[str, Any]], workspace_root: str | Path) -> str:
    """Produce a human-readable diff summary for the given edits."""
    root = Path(workspace_root).resolve()
    lines: list[str] = []
    for e in edits:
        path = e["path"]
        new_text = e["newText"]
        old_text = e.get("oldText")
        if _is_unified_diff(new_text):
            lines.append(f"  {path}: unified diff ({len(new_text)} chars)")
            continue
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
        anchor = e.get("anchor")
        position = (e.get("position") or "after").lower()
        if _is_unified_diff(new_text):
            applied += apply_unified_diff(new_text, root)
            continue
        resolved = (root / path).resolve()
        try:
            if not resolved.is_relative_to(root) and resolved != root:
                continue
        except (ValueError, TypeError):
            continue
        try:
            if old_text is None or old_text == "":
                if anchor:
                    applied += _apply_anchor_edit(resolved, str(anchor), str(new_text), position)
                    continue
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


def _apply_anchor_edit(path: Path, anchor: str, new_text: str, position: str = "after") -> int:
    """Insert new_text before/after the first anchor match. Returns 1 if applied."""
    try:
        current = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    except OSError:
        return 0
    idx = current.find(anchor)
    if idx == -1:
        return 0
    insert_at = idx if position == "before" else idx + len(anchor)
    updated = current[:insert_at] + new_text + current[insert_at:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
        return 1
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Unified diff support
# ---------------------------------------------------------------------------


def _is_unified_diff(text: str) -> bool:
    if not text:
        return False
    return "\n--- " in text and "\n+++ " in text and "\n@@ " in text


def apply_unified_diff(diff_text: str, workspace_root: Path) -> int:
    """Apply a unified diff to files under workspace_root. Returns number of files written."""
    return apply_unified_diff_filtered(diff_text, workspace_root, include_hunk=None)


def apply_unified_diff_filtered(
    diff_text: str,
    workspace_root: Path,
    include_hunk: Callable[[str, int, dict[str, Any]], bool] | None,
) -> int:
    """Apply a unified diff with optional hunk filter.

    include_hunk: callable(path, hunk_index, hunk_dict) -> bool. If None, apply all.
    """
    patches = _parse_unified_diff(diff_text)
    if not patches:
        return 0
    applied = 0
    conflicts: list[str] = []
    for patch in patches:
        rel_path = patch.get("path")
        hunks = patch.get("hunks") or []
        if not rel_path or not hunks:
            continue
        if include_hunk is not None:
            filtered: list[dict[str, Any]] = []
            for i, h in enumerate(hunks):
                try:
                    if include_hunk(rel_path, i, h):
                        filtered.append(h)
                except Exception:
                    continue
            hunks = filtered
            if not hunks:
                continue
        resolved = (workspace_root / rel_path).resolve()
        try:
            if not resolved.is_relative_to(workspace_root) and resolved != workspace_root:
                continue
        except (ValueError, TypeError):
            continue
        try:
            if resolved.exists():
                current = resolved.read_text(encoding="utf-8", errors="replace")
            else:
                current = ""
        except OSError:
            continue
        new_content, ok, hunk_conflicts = _apply_hunks_to_text(current, hunks)
        if not ok:
            if hunk_conflicts:
                conflicts.extend([f"{rel_path}: {c}" for c in hunk_conflicts])
            continue
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(new_content, encoding="utf-8")
            applied += 1
        except OSError:
            continue
    if conflicts:
        _log_edit_conflicts(conflicts)
    return applied


def reverse_unified_diff(diff_text: str) -> str:
    """Best-effort reverse of a unified diff (swap + and -)."""
    lines = diff_text.splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith("--- "):
            out.append(line.replace("--- a/", "--- b/").replace("--- ", "--- "))
            continue
        if line.startswith("+++ "):
            out.append(line.replace("+++ b/", "+++ a/").replace("+++ ", "+++ "))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append("-" + line[1:])
            continue
        if line.startswith("-") and not line.startswith("---"):
            out.append("+" + line[1:])
            continue
        out.append(line)
    return "\n".join(out)


def _parse_unified_diff(diff_text: str) -> list[dict[str, Any]]:
    """Parse a unified diff into a list of file patches."""
    lines = diff_text.splitlines()
    patches: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git"):
            if current:
                patches.append(current)
            current = {"path": None, "hunks": []}
            i += 1
            continue
        if line.startswith("--- "):
            i += 1
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if current is None:
                current = {"path": None, "hunks": []}
            current["path"] = path if path != "/dev/null" else None
            i += 1
            continue
        if line.startswith("@@ "):
            if current is None:
                current = {"path": None, "hunks": []}
            hunk, i = _parse_hunk(lines, i)
            if hunk:
                current["hunks"].append(hunk)
            continue
        i += 1
    if current:
        patches.append(current)
    return patches


def _parse_hunk(lines: list[str], start: int) -> tuple[dict[str, Any] | None, int]:
    """Parse a single @@ hunk. Returns (hunk_dict, next_index)."""
    header = lines[start]
    m = _HUNK_RE.match(header)
    if not m:
        return None, start + 1
    old_start = int(m.group("old_start"))
    old_len = int(m.group("old_len") or "1")
    new_start = int(m.group("new_start"))
    new_len = int(m.group("new_len") or "1")
    hunk_lines: list[tuple[str, str]] = []
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@ "):
            break
        if line.startswith("\\"):
            i += 1
            continue
        if not line:
            hunk_lines.append((" ", ""))
            i += 1
            continue
        tag = line[0]
        if tag not in (" ", "+", "-"):
            break
        hunk_lines.append((tag, line[1:]))
        i += 1
    return (
        {
            "old_start": old_start,
            "old_len": old_len,
            "new_start": new_start,
            "new_len": new_len,
            "lines": hunk_lines,
            "header": header,
        },
        i,
    )


def _apply_hunks_to_text(text: str, hunks: list[dict[str, Any]]) -> tuple[str, bool, list[str]]:
    """Apply hunks to text. Returns (new_text, ok, conflicts)."""
    src_lines = text.splitlines(keepends=True)
    out_lines: list[str] = []
    src_idx = 0
    conflicts: list[str] = []
    for h in hunks:
        old_start = max(1, int(h.get("old_start") or 1))
        hunk_lines = h.get("lines") or []
        target_idx = old_start - 1
        if target_idx < src_idx:
            conflicts.append("hunk target before current cursor")
            return text, False, conflicts
        if not _hunk_matches_at(src_lines, target_idx, hunk_lines):
            fuzzy_idx = _find_hunk_position(src_lines, src_idx, hunk_lines, window=8)
            if fuzzy_idx is None:
                conflicts.append("hunk context not found")
                return text, False, conflicts
            target_idx = fuzzy_idx
        out_lines.extend(src_lines[src_idx:target_idx])
        src_idx = target_idx
        for tag, payload in hunk_lines:
            payload_line = payload + ("\n" if src_idx < len(src_lines) and src_lines[src_idx].endswith("\n") else "")
            if tag == " ":
                if src_idx >= len(src_lines) or src_lines[src_idx].rstrip("\n") != payload:
                    conflicts.append("context line mismatch")
                    return text, False, conflicts
                out_lines.append(src_lines[src_idx])
                src_idx += 1
            elif tag == "-":
                if src_idx >= len(src_lines) or src_lines[src_idx].rstrip("\n") != payload:
                    conflicts.append("deletion line mismatch")
                    return text, False, conflicts
                src_idx += 1
            elif tag == "+":
                out_lines.append(payload_line)
        # Continue to next hunk
    out_lines.extend(src_lines[src_idx:])
    return ("".join(out_lines), True, conflicts)


def _hunk_matches_at(src_lines: list[str], idx: int, hunk_lines: list[tuple[str, str]]) -> bool:
    """Check if hunk context matches starting at idx (only ' ' and '-' lines)."""
    i = idx
    for tag, payload in hunk_lines:
        if tag not in (" ", "-"):
            continue
        if i >= len(src_lines):
            return False
        if src_lines[i].rstrip("\n") != payload:
            return False
        i += 1
    return True


def _find_hunk_position(
    src_lines: list[str],
    start_idx: int,
    hunk_lines: list[tuple[str, str]],
    window: int = 5,
) -> int | None:
    """Find a nearby index where hunk context matches."""
    lo = max(0, start_idx - window)
    hi = min(len(src_lines), start_idx + window)
    for idx in range(lo, hi + 1):
        if _hunk_matches_at(src_lines, idx, hunk_lines):
            return idx
    return None


def _log_edit_conflicts(conflicts: list[str]) -> None:
    """Best-effort conflict log for failed unified diff application."""
    try:
        from pathlib import Path

        log_path = Path.home() / ".ollamacode" / "edit_conflicts.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for c in conflicts:
                f.write(c[:400] + "\n")
    except OSError:
        pass


_HUNK_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_len>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))?\s+@@"
)
