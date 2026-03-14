"""Export/Import session dialogs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


class ExportDialog(ModalScreen[None]):
    """Show exported session JSON with copy button."""

    DEFAULT_CSS = """
    ExportDialog {
        align: center middle;
    }
    #export-dialog {
        width: 80;
        max-height: 80%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    #export-text {
        height: 20;
    }
    """

    def __init__(self, json_text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._json_text = json_text

    def compose(self) -> ComposeResult:
        with Vertical(id="export-dialog"):
            yield Static("Exported Session JSON", classes="dialog-title")
            yield TextArea(self._json_text, id="export-text", read_only=True)
            yield Button("Copy to Clipboard", variant="primary", id="export-copy")
            yield Button("Close", variant="default", id="export-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-copy":
            import subprocess

            try:
                subprocess.run(
                    ["pbcopy"],
                    input=self._json_text.encode(),
                    check=True,
                    timeout=5,
                )
                self.app.notify("Copied to clipboard")
            except Exception:
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=self._json_text.encode(),
                        check=True,
                        timeout=5,
                    )
                    self.app.notify("Copied to clipboard")
                except Exception:
                    self.app.notify("Clipboard not available", severity="warning")
        elif event.button.id == "export-close":
            self.dismiss(None)


class ImportDialog(ModalScreen[str]):
    """Paste session JSON to import."""

    DEFAULT_CSS = """
    ImportDialog {
        align: center middle;
    }
    #import-dialog {
        width: 80;
        max-height: 80%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    #import-text {
        height: 15;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="import-dialog"):
            yield Static("Paste Session JSON", classes="dialog-title")
            yield TextArea("", id="import-text", language="json")
            yield Button("Import", variant="primary", id="import-ok")
            yield Button("Cancel", variant="default", id="import-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "import-ok":
            text_area = self.query_one("#import-text", TextArea)
            self.dismiss(text_area.text)
        elif event.button.id == "import-cancel":
            self.dismiss("")
