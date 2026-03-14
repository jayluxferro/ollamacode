"""Agent mode picker dialog — switch between Build and Plan modes."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


AGENT_MODES = [
    {
        "name": "build",
        "label": "Build",
        "description": "Full-access development agent. All tools enabled.",
        "color": "green",
    },
    {
        "name": "plan",
        "label": "Plan",
        "description": "Read-only analysis agent. File edits require approval.",
        "color": "blue",
    },
]


class AgentPickerDialog(ModalScreen[str]):
    """Modal for switching between agent modes."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("1", "select_build", "Build", show=False),
        Binding("2", "select_plan", "Plan", show=False),
    ]

    def __init__(self, current: str = "build") -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-picker-container"):
            yield Static("[bold]Select Agent Mode[/]")
            for mode in AGENT_MODES:
                marker = " \u2713" if mode["name"] == self._current else ""
                yield Button(
                    f"{mode['label']}{marker} — {mode['description']}",
                    id=f"agent-{mode['name']}",
                    variant="success" if mode["name"] == self._current else "default",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("agent-"):
            self.dismiss(bid.removeprefix("agent-"))

    def action_select_build(self) -> None:
        self.dismiss("build")

    def action_select_plan(self) -> None:
        self.dismiss("plan")

    def action_cancel(self) -> None:
        self.dismiss("")
