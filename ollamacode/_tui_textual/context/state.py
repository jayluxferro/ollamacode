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
    agent_mode: str = "build"  # "build", "plan", or "review"
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
    compact_mode: str = "off"  # "on", "off", "auto"
    trace_filter: str = ""
    variant_name: str = ""
    checkpoint_count: int = 0


@dataclass
class AppState:
    """Application-wide state shared across sessions."""

    workspace_root: str = "."
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    lsp_servers: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[dict[str, Any]] = field(default_factory=list)
    message_history: list[dict[str, Any]] = field(default_factory=list)
    # Manager references (set in app.__init__)
    permissions_manager: Any = None
    permission_state: Any = None
    mode_manager: Any = None
    command_manager: Any = None
    variant_manager: Any = None
    plugin_manager: Any = None
    file_watcher_handle: Any = None
