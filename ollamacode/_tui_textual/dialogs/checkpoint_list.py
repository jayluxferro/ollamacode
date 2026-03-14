"""Checkpoint list dialog — browse and restore checkpoints."""

from __future__ import annotations

from functools import partial

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class CheckpointListDialog(ModalScreen[str]):
    """Browse checkpoints and select one to restore."""

    DEFAULT_CSS = """
    CheckpointListDialog {
        align: center middle;
    }
    #checkpoint-dialog {
        width: 70;
        max-height: 80%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    .checkpoint-row {
        height: 3;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    .checkpoint-row:hover {
        background: $accent 20%;
    }
    """

    def __init__(self, session_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._checkpoints: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="checkpoint-dialog"):
            yield Static("Checkpoints", classes="dialog-title")
            yield Vertical(id="checkpoint-list")
            yield Button("Cancel", variant="default", id="checkpoint-cancel")

    def on_mount(self) -> None:
        try:
            from ollamacode.checkpoints import list_checkpoints

            self._checkpoints = list_checkpoints(self._session_id)
        except Exception:
            self._checkpoints = []

        container = self.query_one("#checkpoint-list", Vertical)
        if not self._checkpoints:
            container.mount(Static("No checkpoints found", classes="sidebar-muted"))
            return

        for cp in self._checkpoints:
            cp_id = cp.get("id", "")
            prompt = cp.get("prompt", "")[:50]
            file_count = cp.get("file_count", 0)
            created = cp.get("created_at", "")[:19]
            label = f"{cp_id[:8]}  {file_count} files  {created}\n  {prompt}"
            btn = Button(label, id=f"cp-{cp_id}", classes="checkpoint-row")
            btn._checkpoint_id = cp_id  # type: ignore[attr-defined]
            container.mount(btn)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "checkpoint-cancel":
            self.dismiss("")
        elif event.button.id and event.button.id.startswith("cp-"):
            cp_id = getattr(event.button, "_checkpoint_id", "")
            self.dismiss(cp_id)
