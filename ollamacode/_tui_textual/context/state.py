"""Reactive state definitions for the OllamaCode Textual TUI.

Provides two dataclasses that hold the full UI-visible state:

* :class:`SessionState` -- per-session data (model, tokens, tool calls, ...).
* :class:`AppState` -- application-wide data (workspace, MCP servers, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """State scoped to a single chat session."""

    session_id: str = ""
    title: str = ""
    model: str = ""
    provider_name: str = "ollama"
    agent_mode: str = "build"  # "build" or "plan"
    token_count: int = 0
    cost: float = 0.0
    context_limit: int = 128_000
    is_busy: bool = False
    is_streaming: bool = False
    tool_calls: int = 0
    tool_errors: int = 0
    current_tool: str = ""
    permissions_granted: int = 0
    autonomous: bool = False


@dataclass
class AppState:
    """Application-wide state shared across sessions."""

    workspace_root: str = "."
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    lsp_servers: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[dict[str, Any]] = field(default_factory=list)
    message_history: list[dict[str, Any]] = field(default_factory=list)
