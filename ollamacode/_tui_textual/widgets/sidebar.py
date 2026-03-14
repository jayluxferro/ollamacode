"""Sidebar with collapsible sections for session context."""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static
from textual.reactive import reactive

logger = logging.getLogger(__name__)


class Sidebar(VerticalScroll):
    """Right sidebar with session context, MCP status, TODOs, and modified files."""

    # Reactive state
    session_title = reactive("New Session")
    token_count = reactive(0)
    context_limit = reactive(128000)
    cost = reactive(0.0)
    mcp_servers: reactive[list[dict[str, Any]]] = reactive(list, always_update=True)
    lsp_servers: reactive[list[dict[str, Any]]] = reactive(list, always_update=True)
    todos: reactive[list[dict[str, Any]]] = reactive(list, always_update=True)
    modified_files: reactive[list[dict[str, Any]]] = reactive(list, always_update=True)
    permissions_granted = reactive(0)
    permissions_denied = reactive(0)
    agent_mode = reactive("build")
    checkpoint_count = reactive(0)
    plugin_count = reactive(0)

    def compose(self) -> ComposeResult:
        # Session section
        with Vertical(classes="sidebar-section"):
            yield Static("SESSION", classes="sidebar-section-title")
            yield Static(
                self.session_title, id="sidebar-session-title", classes="sidebar-item"
            )

        # Context section
        with Vertical(classes="sidebar-section"):
            yield Static("CONTEXT", classes="sidebar-section-title")
            yield Static(
                "0 / 128K tokens",
                id="sidebar-context-tokens",
                classes="sidebar-item sidebar-muted",
            )
            yield Static(
                "$0.0000",
                id="sidebar-context-cost",
                classes="sidebar-item sidebar-muted",
            )

        # Agent Mode section
        with Vertical(classes="sidebar-section", id="sidebar-mode-section"):
            yield Static("AGENT MODE", classes="sidebar-section-title")
            yield Static(
                "BUILD",
                id="sidebar-mode-value",
                classes="sidebar-item",
            )

        # Permissions section
        with Vertical(classes="sidebar-section", id="sidebar-permissions-section"):
            yield Static("PERMISSIONS", classes="sidebar-section-title")
            yield Static(
                "0 granted / 0 denied",
                id="sidebar-permissions-value",
                classes="sidebar-item sidebar-muted",
            )

        # MCP Servers section
        with Vertical(classes="sidebar-section", id="sidebar-mcp-section"):
            yield Static("MCP SERVERS", classes="sidebar-section-title")
            yield Static(
                "No servers connected",
                id="sidebar-mcp-list",
                classes="sidebar-item sidebar-muted",
            )

        # LSP Servers section
        with Vertical(classes="sidebar-section", id="sidebar-lsp-section"):
            yield Static("LSP SERVERS", classes="sidebar-section-title")
            yield Static(
                "None active",
                id="sidebar-lsp-list",
                classes="sidebar-item sidebar-muted",
            )

        # Checkpoints section
        with Vertical(classes="sidebar-section", id="sidebar-checkpoints-section"):
            yield Static("CHECKPOINTS", classes="sidebar-section-title")
            yield Static(
                "0 checkpoints",
                id="sidebar-checkpoints-value",
                classes="sidebar-item sidebar-muted",
            )

        # Plugins section
        with Vertical(classes="sidebar-section", id="sidebar-plugins-section"):
            yield Static("PLUGINS", classes="sidebar-section-title")
            yield Static(
                "0 loaded",
                id="sidebar-plugins-value",
                classes="sidebar-item sidebar-muted",
            )

        # TODO section
        with Vertical(classes="sidebar-section", id="sidebar-todo-section"):
            yield Static("TODO", classes="sidebar-section-title")
            yield Static(
                "No tasks", id="sidebar-todo-list", classes="sidebar-item sidebar-muted"
            )

        # Modified Files section
        with Vertical(classes="sidebar-section", id="sidebar-files-section"):
            yield Static("MODIFIED FILES", classes="sidebar-section-title")
            yield Static(
                "No changes",
                id="sidebar-files-list",
                classes="sidebar-item sidebar-muted",
            )

    def watch_session_title(self, value: str) -> None:
        try:
            self.query_one("#sidebar-session-title", Static).update(value[:40])
        except Exception:
            pass

    def watch_token_count(self, value: int) -> None:
        try:
            limit_k = self.context_limit / 1000
            if value >= 1000:
                display = f"{value / 1000:.1f}K / {limit_k:.0f}K tokens"
            else:
                display = f"{value} / {limit_k:.0f}K tokens"
            pct = min(100, int(value / max(1, self.context_limit) * 100))
            self.query_one("#sidebar-context-tokens", Static).update(
                f"{display} ({pct}%)"
            )
        except Exception:
            pass

    def watch_cost(self, value: float) -> None:
        try:
            self.query_one("#sidebar-context-cost", Static).update(f"${value:.4f}")
        except Exception:
            pass

    def watch_mcp_servers(self, value: list[dict[str, Any]]) -> None:
        self.update_mcp_servers(value)

    def watch_lsp_servers(self, value: list[dict[str, Any]]) -> None:
        self.update_lsp_servers(value)

    def watch_todos(self, value: list[dict[str, Any]]) -> None:
        self.update_todos(value)

    def watch_modified_files(self, value: list[dict[str, Any]]) -> None:
        self.update_modified_files(value)

    def watch_agent_mode(self, value: str) -> None:
        try:
            self.query_one("#sidebar-mode-value", Static).update(value.upper())
        except Exception:
            pass

    def watch_permissions_granted(self, value: int) -> None:
        self._update_permissions_display()

    def watch_permissions_denied(self, value: int) -> None:
        self._update_permissions_display()

    def _update_permissions_display(self) -> None:
        try:
            self.query_one("#sidebar-permissions-value", Static).update(
                f"{self.permissions_granted} granted / {self.permissions_denied} denied"
            )
        except Exception:
            pass

    def watch_checkpoint_count(self, value: int) -> None:
        try:
            self.query_one("#sidebar-checkpoints-value", Static).update(
                f"{value} checkpoint{'s' if value != 1 else ''}"
            )
        except Exception:
            pass

    def watch_plugin_count(self, value: int) -> None:
        try:
            self.query_one("#sidebar-plugins-value", Static).update(
                f"{value} loaded"
            )
        except Exception:
            pass

    def update_mcp_servers(self, servers: list[dict[str, Any]]) -> None:
        """Update MCP server list display."""
        try:
            widget = self.query_one("#sidebar-mcp-list", Static)
            if not servers:
                widget.update("No servers connected")
                return
            lines = []
            for s in servers:
                name = s.get("name", "unknown")
                status = s.get("status", "connected")
                dot = "\u25cf" if status == "connected" else "\u25cb"
                color = "green" if status == "connected" else "red"
                lines.append(f"[{color}]{dot}[/] {name}")
            widget.update("\n".join(lines))
        except Exception:
            pass

    def update_lsp_servers(self, servers: list[dict[str, Any]]) -> None:
        """Update LSP server list display."""
        try:
            widget = self.query_one("#sidebar-lsp-list", Static)
            if not servers:
                widget.update("None active")
                return
            lines = []
            for server in servers:
                name = server.get("name", "unknown")
                status = server.get("status", "connected")
                dot = "\u25cf" if status == "connected" else "\u25cb"
                color = "green" if status == "connected" else "red"
                lines.append(f"[{color}]{dot}[/] {name}")
            widget.update("\n".join(lines[:10]))
        except Exception:
            pass

    def update_todos(self, todos: list[dict[str, Any]]) -> None:
        """Update TODO list display."""
        try:
            widget = self.query_one("#sidebar-todo-list", Static)
            if not todos:
                widget.update("No tasks")
                return
            lines = []
            for t in todos:
                status = str(
                    t.get("status") or ("completed" if t.get("done") else "pending")
                ).lower()
                text = t.get("content") or t.get("text", "")
                priority = str(t.get("priority") or "").lower()
                if status == "completed":
                    prefix = "[green]\u2611[/]"
                elif status == "in_progress":
                    prefix = "[yellow]\u25d0[/]"
                elif status == "cancelled":
                    prefix = "[red]\u2298[/]"
                else:
                    prefix = "\u2610"
                if priority == "high":
                    text = f"[bold]{text}[/]"
                elif status == "completed":
                    text = f"[dim]{text}[/]"
                lines.append(f"{prefix} {text}")
            widget.update("\n".join(lines[:10]))
        except Exception:
            pass

    def update_modified_files(self, files: list[dict[str, Any]]) -> None:
        """Update modified files display."""
        try:
            widget = self.query_one("#sidebar-files-list", Static)
            if not files:
                widget.update("No changes")
                return
            lines = []
            for f in files:
                path = f.get("path", "")
                added = f.get("added", 0)
                removed = f.get("removed", 0)
                event = str(f.get("event", "")).lower()
                if added or removed:
                    lines.append(f"[green]+{added}[/] [red]-{removed}[/] {path}")
                elif event == "created":
                    lines.append(f"[green]new[/] {path}")
                elif event == "deleted":
                    lines.append(f"[red]del[/] {path}")
                elif event:
                    lines.append(f"[yellow]{event[:3]}[/] {path}")
                else:
                    lines.append(path)
            widget.update("\n".join(lines[:15]))
        except Exception:
            pass
