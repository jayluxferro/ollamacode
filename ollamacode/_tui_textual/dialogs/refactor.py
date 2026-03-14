"""Refactor dialog — choose refactoring operation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class RefactorDialog(ModalScreen[str]):
    """Choose a refactoring operation."""

    DEFAULT_CSS = """
    RefactorDialog {
        align: center middle;
    }
    #refactor-dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    """

    OPERATIONS = [
        ("index", "Index Symbols", "Index all symbols in workspace"),
        ("rename", "Rename Symbol", "Rename a symbol across files"),
        ("extract", "Extract Function", "Extract selection into a function"),
        ("move", "Move Symbol", "Move a symbol to another file"),
        ("rollback", "Rollback Changes", "Undo last refactoring"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="refactor-dialog"):
            yield Static("Refactoring", classes="dialog-title")
            for op_id, label, desc in self.OPERATIONS:
                yield Button(f"{label} — {desc}", id=f"refactor-{op_id}")
            yield Button("Cancel", variant="default", id="refactor-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refactor-cancel":
            self.dismiss("")
        elif event.button.id and event.button.id.startswith("refactor-"):
            op = event.button.id.removeprefix("refactor-")
            self.dismiss(op)
