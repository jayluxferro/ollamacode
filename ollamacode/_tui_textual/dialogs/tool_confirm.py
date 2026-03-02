"""Tool confirmation dialog — allow/always/deny tool execution."""

from __future__ import annotations

import json
import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

logger = logging.getLogger(__name__)

# Tool icons
TOOL_ICONS: dict[str, str] = {
    "run_command": "$",
    "bash": "$",
    "read_file": "\u2192",
    "write_file": "\u2190",
    "edit_file": "\u270e",
    "multi_edit": "\u270e",
    "search_codebase": "\u25c7",
    "grep": "\u2731",
    "glob": "\u2731",
    "apply_patch": "\u229e",
    "ask_user": "?",
    "fetch_url": "%",
    "web_search": "\u25c8",
}


class ToolConfirmDialog(ModalScreen[str]):
    """Modal dialog for confirming tool execution."""

    BINDINGS = [
        Binding("y", "allow", "Allow", show=True),
        Binding("a", "always_allow", "Always", show=True),
        Binding("n", "deny", "Deny", show=True),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, tool_name: str, args: dict[str, Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = args

    def compose(self) -> ComposeResult:
        icon = TOOL_ICONS.get(self.tool_name, "\u2699")
        with Vertical(id="tool-confirm-container"):
            yield Static(
                f"{icon} Execute tool: [bold]{self.tool_name}[/bold]",
                id="tool-confirm-title",
            )
            # Format args
            try:
                args_text = json.dumps(self.tool_args, indent=2, default=str)
            except Exception:
                args_text = str(self.tool_args)
            if len(args_text) > 2000:
                args_text = args_text[:2000] + "\n..."
            yield Static(args_text, id="tool-confirm-args")
            with Horizontal(id="tool-confirm-buttons"):
                yield Button("Allow (y)", variant="success", id="btn-allow")
                yield Button("Always (a)", variant="primary", id="btn-always")
                yield Button("Deny (n)", variant="error", id="btn-deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-allow":
            self.dismiss("allow")
        elif event.button.id == "btn-always":
            self.dismiss("always")
        elif event.button.id == "btn-deny":
            self.dismiss("deny")

    def action_allow(self) -> None:
        self.dismiss("allow")

    def action_always_allow(self) -> None:
        self.dismiss("always")

    def action_deny(self) -> None:
        self.dismiss("deny")
