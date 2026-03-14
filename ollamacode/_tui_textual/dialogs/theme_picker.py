"""Theme picker dialog — select a color theme."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from ..context.theme import get_theme, list_themes


class ThemePickerDialog(ModalScreen[str]):
    """Modal for selecting a color theme."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, current: str = "opencode") -> None:
        super().__init__()
        self._current = current
        self._themes = list_themes()
        self._selected_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-picker-container"):
            yield Static("[bold]Select Theme[/]")
            with VerticalScroll(id="theme-options"):
                for i, name in enumerate(self._themes):
                    theme = get_theme(name)
                    primary = theme.get("primary", "#ffffff")
                    marker = " \u2713" if name == self._current else ""
                    cls = "theme-option" + (" -selected" if i == 0 else "")
                    yield Static(
                        f"[{primary}]\u2588\u2588[/] {name}{marker}",
                        classes=cls,
                        name=name,
                    )

    def key_up(self) -> None:
        self._move(-1)

    def key_down(self) -> None:
        self._move(1)

    def key_enter(self) -> None:
        if self._themes and 0 <= self._selected_index < len(self._themes):
            self.dismiss(self._themes[self._selected_index])

    def _move(self, delta: int) -> None:
        options = list(self.query(".theme-option"))
        if not options:
            return
        if 0 <= self._selected_index < len(options):
            options[self._selected_index].remove_class("-selected")
        self._selected_index = max(
            0, min(len(options) - 1, self._selected_index + delta)
        )
        options[self._selected_index].add_class("-selected")
        options[self._selected_index].scroll_visible()

    def on_click(self, event) -> None:
        name = getattr(event.widget, "name", None)
        if name and name in self._themes:
            self.dismiss(name)

    def action_cancel(self) -> None:
        self.dismiss("")
