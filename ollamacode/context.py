"""
Context injection: @-style file/folder references and --file/--lines in the user message.

Parse @path or @path/to/dir/ in the message; inject file contents or folder listing.
prepend_file_context: add file (or line range) to the start of the prompt for chat-with-selection.
get_branch_context: git diff against base branch and optional PR description for system prompt.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CONTEXT_FILE_SIZE = 1_048_576  # 1 MB


def get_branch_summary_one_line(
    workspace_root: str | Path,
    base_branch: str = "main",
) -> str:
    """
    Return a one-line summary for the system prompt: current branch and last commit.
    Returns "" if not a git repo or on error.
    """
    root = Path(workspace_root).resolve()
    parts: list[str] = []
    try:
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if branch_result.returncode == 0 and branch_result.stdout.strip():
            parts.append("Branch: " + branch_result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    try:
        log_result = subprocess.run(
            ["git", "log", "-1", "--oneline", "--no-decorate"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if log_result.returncode == 0 and log_result.stdout.strip():
            parts.append("Last commit: " + log_result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    if not parts:
        return ""
    return " ".join(parts)


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
                    "--- PR / change description ---\n"
                    + pr_path.read_text(encoding="utf-8", errors="replace").strip()
                )
        except (ValueError, TypeError, OSError):
            pass
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def load_ollama_md_context(workspace_root: str | Path) -> str:
    """
    Load user and project OLLAMA.md / CLAUDE.md context for the system prompt.
    User: ~/.ollamacode/OLLAMA.md and ~/.ollamacode/CLAUDE.md (if present).
    Project: .ollamacode/OLLAMA.md or OLLAMA.md; .ollamacode/CLAUDE.md or CLAUDE.md.
    Returns a single string to append to the system prompt, or "" if none exist.
    """
    root = Path(workspace_root).resolve()
    parts: list[str] = []

    def _maybe_add(path: Path, label: str) -> None:
        try:
            if path.is_file():
                parts.append(
                    f"--- {label} ---\n"
                    + path.read_text(encoding="utf-8", errors="replace").strip()
                )
        except OSError as exc:
            logger.warning("Failed to read context file %s: %s", path, exc)

    user_dir = Path.home() / ".ollamacode"
    _maybe_add(user_dir / "OLLAMA.md", "User context (OLLAMA.md)")
    _maybe_add(user_dir / "CLAUDE.md", "User context (CLAUDE.md)")

    for name in (".ollamacode/OLLAMA.md", "OLLAMA.md"):
        proj_path = root / name
        if proj_path.is_file():
            _maybe_add(proj_path, "Project context (OLLAMA.md)")
            break

    for name in (".ollamacode/CLAUDE.md", "CLAUDE.md"):
        proj_path = root / name
        if proj_path.is_file():
            _maybe_add(proj_path, "Project context (CLAUDE.md)")
            break

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
        file_size = resolved.stat().st_size
        if file_size > _MAX_CONTEXT_FILE_SIZE:
            logger.warning(
                "File %s too large for context (%d bytes, limit %d); skipping",
                resolved,
                file_size,
                _MAX_CONTEXT_FILE_SIZE,
            )
            return message
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


def _fuzzy_find_file(root: Path, name: str, max_results: int = 5) -> list[Path]:
    """Search for files in root whose name contains the given string (case-insensitive).
    Returns up to max_results matches, preferring exact name matches first."""
    needle = name.lower()
    exact: list[Path] = []
    partial: list[Path] = []
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            pname = p.name.lower()
            if pname == needle:
                exact.append(p)
            elif needle in pname:
                partial.append(p)
            if len(exact) + len(partial) >= max_results * 3:
                break
    except OSError:
        pass
    results = exact + partial
    return results[:max_results]


def expand_at_refs(message: str, workspace_root: str | Path) -> str:
    """
    Expand @path and @folder/ references in the message.
    Replaces each @path with injected content (file contents or folder file list)
    and prepends the injected block to the message.
    When an exact path is not found, performs fuzzy filename search.
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
                file_size = resolved.stat().st_size
                if file_size > _MAX_CONTEXT_FILE_SIZE:
                    logger.warning(
                        "File %s too large for @-ref context (%d bytes, limit %d); skipping",
                        resolved,
                        file_size,
                        _MAX_CONTEXT_FILE_SIZE,
                    )
                    continue
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Failed to read @-ref file %s: %s", resolved, exc)
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
        else:
            # Fuzzy search: file/dir not found at exact path — search by filename
            basename = Path(ref).name
            if not basename or len(basename) < 2:
                continue
            fuzzy_matches = _fuzzy_find_file(root, basename, max_results=1)
            if fuzzy_matches:
                found = fuzzy_matches[0]
                try:
                    file_size = found.stat().st_size
                    if file_size > _MAX_CONTEXT_FILE_SIZE:
                        continue
                    content = found.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel_path = str(found.relative_to(root))
                injected.append(
                    f"Contents of {rel_path} (fuzzy match for @{ref}):\n```\n{content}\n```"
                )
                new_message = new_message.replace(match.group(0), "", 1)
                logger.debug("Fuzzy @-ref: %s -> %s", ref, rel_path)

    if not injected:
        return message
    return "\n\n".join(injected) + "\n\n" + new_message.strip()
