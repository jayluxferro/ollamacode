"""OllamaCode TUI — Textual-based terminal interface.

Public API:
    run_tui()  — async entry point matching the old tui.py signature
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
McpConnection = Any  # type alias for mcp session

# Regex for stripping ANSI escape sequences
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _sanitize_stream_text(text: str) -> str:
    """Remove terminal control noise from streaming text.

    Strips ANSI escape sequences, carriage returns, and null bytes
    so TUI panels don't show raw control characters.
    """
    if not text:
        return ""
    cleaned = _ANSI_ESCAPE_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "")
    return cleaned


async def run_tui(
    session: Any | None,
    model: str,
    system_extra: str,
    *,
    quiet: bool = False,
    max_tool_rounds: int = 20,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    timing: bool = False,
    workspace_root: str | None = None,
    linter_command: str | None = None,
    test_command: str | None = None,
    docs_command: str | None = None,
    profile_command: str | None = None,
    show_semantic_hint: bool = False,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
    branch_context: bool = False,
    branch_context_base: str = "main",
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    permissions_config: dict[str, Any] | None = None,
    code_style: str | None = None,
    planner_model: str | None = None,
    executor_model: str | None = None,
    reviewer_model: str | None = None,
    multi_agent_max_iterations: int = 2,
    multi_agent_require_review: bool = True,
    tui_tool_trace_max: int = 20,
    tui_tool_log_max: int = 8,
    tui_tool_log_chars: int = 160,
    tui_refresh_hz: int = 12,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
    autonomous_mode: bool = False,
    subagents: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    session_title: str | None = None,
    session_history: list[dict[str, Any]] | None = None,
    provider: Any = None,
    provider_name: str = "ollama",
) -> None:
    """Run the interactive Textual TUI.

    This function signature matches the old Rich-based tui.py for backwards
    compatibility. All parameters are forwarded to OllamaCodeApp.
    """
    # Suppress noisy logs
    for _name in (
        "mcp",
        "mcp.client",
        "mcp.server",
        "mcp.server.lowlevel",
        "mcp.server.lowlevel.server",
        "httpx",
        "urllib3",
    ):
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.WARNING)
        _lg.propagate = False

    # Determine theme from environment
    theme_name = os.environ.get("OLLAMACODE_THEME", "opencode")

    # Build config dict from yaml if available
    config: dict[str, Any] = {}
    try:
        from ollamacode.config import load_config
        from pathlib import Path

        config = load_config(cwd=Path(workspace_root) if workspace_root else None)
    except Exception:
        pass

    # Merge runtime params into config for manager init
    if subagents:
        config["subagents"] = subagents
    if permissions_config is not None:
        config.setdefault("permissions", {}).update(permissions_config)

    from .app import OllamaCodeApp

    app = OllamaCodeApp(
        session=session,
        model=model,
        system_extra=system_extra,
        workspace_root=workspace_root,
        provider=provider,
        provider_name=provider_name,
        theme_name=theme_name,
        session_id=session_id,
        session_title=session_title,
        session_history=session_history,
        confirm_tool_calls=confirm_tool_calls,
        autonomous_mode=autonomous_mode,
        max_tool_rounds=max_tool_rounds,
        max_messages=max_messages,
        max_tool_result_chars=max_tool_result_chars,
        linter_command=linter_command,
        test_command=test_command,
        docs_command=docs_command,
        profile_command=profile_command,
        config=config,
        allowed_tools=allowed_tools,
        blocked_tools=blocked_tools,
    )

    await app.run_async()
