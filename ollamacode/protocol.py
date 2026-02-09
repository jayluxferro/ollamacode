"""
Structured editor protocol: request/response shapes for chat-with-selection and apply-edits.

See docs/STRUCTURED_PROTOCOL.md for the full spec.
"""

from __future__ import annotations

from typing import Any


def normalize_chat_body(body: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """
    Normalize a protocol chat request body to (message, file_path, lines_spec).

    Accepts:
      - message (required)
      - file (optional), lines (optional) — line range "start-end" or "start:end" (1-based)
      - selection (optional) — { "file": str, "startLine": int, "endLine": int } (1-based inclusive)
    If both file/lines and selection are present, selection wins.
    """
    message = (body.get("message") or "").strip()
    file_path = body.get("file")
    lines_spec = body.get("lines")
    selection = body.get("selection")
    if isinstance(selection, dict):
        sel_file = selection.get("file")
        start = selection.get("startLine")
        end = selection.get("endLine")
        if isinstance(sel_file, str) and sel_file.strip():
            file_path = sel_file.strip()
            if isinstance(start, int) and isinstance(end, int):
                lines_spec = f"{start}-{end}"
            elif isinstance(start, int):
                lines_spec = str(start)
    return message, file_path if file_path else None, lines_spec if lines_spec else None
