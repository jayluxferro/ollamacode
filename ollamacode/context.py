"""
Context injection: @-style file/folder references and --file/--lines in the user message.

Parse @path or @path/to/dir/ in the message; inject file contents or folder listing.
prepend_file_context: add file (or line range) to the start of the prompt for chat-with-selection.
get_branch_context: git diff against base branch and optional PR description for system prompt.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def get_branch_context(
    workspace_root: str | Path,
    base_branch: str = "main",
    pr_description_file: str | None = None,
) -> str:
    """
    Return a string to append to the system prompt: git diff against base_branch and optionally
    PR description from a file. Returns "" if not a git repo or on error.
    """
    root = Path(workspace_root).resolve()
    parts: list[str] = []
    try:
        result = subprocess.run(
            ["git", "diff", base_branch, "--", "."],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(f"--- Git diff vs {base_branch} ---\n" + result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    if pr_description_file:
        pr_path = (root / pr_description_file).resolve()
        try:
            if pr_path.is_relative_to(root) and pr_path.is_file():
                parts.append(
                    "--- PR / change description ---\n" + pr_path.read_text(encoding="utf-8", errors="replace").strip()
                )
        except (ValueError, TypeError, OSError):
            pass
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def prepend_file_context(
    message: str,
    file_path: str,
    workspace_root: str | Path,
    lines_spec: str | None = None,
) -> str:
    """
    Prepend file contents (optionally a line range) to the message.
    lines_spec: "START-END" or "START:END" (1-based inclusive). If None, entire file.
    Path is resolved relative to workspace_root. Returns message unchanged if file missing.
    """
    root = Path(workspace_root).resolve()
    resolved = (root / file_path).resolve()
    try:
        if not resolved.is_relative_to(root) and resolved != root:
            return message
    except (ValueError, TypeError):
        return message
    if not resolved.is_file():
        return message
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return message
    if lines_spec:
        parts = re.split(r"[-:]", lines_spec.strip(), maxsplit=1)
        try:
            start = max(1, int(parts[0].strip()))
            end = int(parts[1].strip()) if len(parts) > 1 else start
            if end < start:
                start, end = end, start
            lines = content.splitlines()
            # 1-based inclusive -> 0-based slice
            start0 = max(0, start - 1)
            end0 = min(len(lines), end)
            content = "\n".join(lines[start0:end0])
            label = (
                f"Contents of {file_path} (lines {start}-{end}):"
                if end != start
                else f"Contents of {file_path} (line {start}):"
            )
        except (ValueError, IndexError):
            label = f"Contents of {file_path}:"
    else:
        label = f"Contents of {file_path}:"
    return f"{label}\n```\n{content}\n```\n\n{message.strip()}"


def expand_at_refs(message: str, workspace_root: str | Path) -> str:
    """
    Expand @path and @folder/ references in the message.
    Replaces each @path with injected content (file contents or folder file list)
    and prepends the injected block to the message.
    """
    root = Path(workspace_root).resolve()
    # Match @path (path = non-whitespace, may contain /). Greedy so @main.py gets "main.py".
    pattern = re.compile(r"@([^\s@][^\s]*)")
    injected: list[str] = []
    seen: set[str] = set()
    new_message = message

    for match in pattern.finditer(message):
        ref = match.group(1).rstrip("/")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        resolved = (root / ref).resolve()
        try:
            if not resolved.is_relative_to(root) and resolved != root:
                continue  # outside workspace, skip
        except (ValueError, TypeError):
            continue
        if resolved.is_file():
            try:
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            injected.append(f"Contents of {ref}:\n```\n{content}\n```")
            new_message = new_message.replace(match.group(0), "", 1)
        elif resolved.is_dir():
            try:
                entries = sorted(resolved.iterdir())
            except OSError:
                continue
            # Limit to 100 entries, names only
            names = [e.name for e in entries[:100]]
            if len(entries) > 100:
                names.append("...")
            injected.append(f"Files in {ref}/:\n" + "\n".join(names))
            new_message = new_message.replace(match.group(0), "", 1)

    if not injected:
        return message
    return "\n\n".join(injected) + "\n\n" + new_message.strip()
