"""
IDE diagnostics: run linter and return LSP-like diagnostics (path, range, message, severity).

Used by ollamacode/diagnostics protocol and POST /diagnostics for editor integration.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _parse_ruff_or_generic(output: str, workspace_root: str) -> list[dict]:
    """
    Parse linter output into LSP-like diagnostics.
    Supports: file:line:col: message, file:line: message, and ruff format.
    """
    diagnostics: list[dict] = []
    root = Path(workspace_root).resolve()
    # file:line:col: code message  or  file:line: message
    pattern = re.compile(
        r"^(.+?):(\d+)(?::(\d+))?\s*:?\s*(.*)$",
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        path_str, line_str, col_str, rest = m.groups()
        path = Path(path_str).resolve()
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
        line = max(1, int(line_str))
        col = max(1, int(col_str)) if col_str else 1
        message = (rest or "").strip()
        if not message:
            continue
        diagnostics.append(
            {
                "path": str(path),
                "range": {
                    "start": {"line": line - 1, "character": col - 1},
                    "end": {"line": line - 1, "character": col - 1},
                },
                "message": message[:500],
                "severity": "warning"
                if "warning" in message.lower()
                or "w" == (message.split()[0] if message.split() else "")
                else "error",
            }
        )
    return diagnostics


def get_diagnostics(
    workspace_root: str,
    path: str | None = None,
    linter_command: str = "ruff check .",
    timeout_seconds: int = 30,
) -> list[dict]:
    """
    Run linter and return list of LSP-like diagnostics.
    Each item: { path, range: { start: { line, character }, end }, message, severity }.
    """
    cmd = linter_command if linter_command else "ruff check ."
    if path:
        cmd = f"{cmd} {path!r}" if path not in cmd else cmd
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        out = (result.stdout or "") + (result.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    return _parse_ruff_or_generic(out, workspace_root)
