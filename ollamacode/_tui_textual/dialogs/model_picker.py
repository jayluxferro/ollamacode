"""Model picker dialog — select an LLM model."""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

logger = logging.getLogger(__name__)


class ModelPickerDialog(ModalScreen[str]):
    """Modal for selecting a model from available options."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, models: list[str] | None = None, current: str = "") -> None:
        super().__init__()
        self._models = models or []
        self._current = current
        self._filtered: list[str] = []
        self._selected_index: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-container"):
            yield Static("[bold]Select Model[/]")
            yield Input(placeholder="Filter models...", id="model-search")
            yield VerticalScroll(id="model-options")

    def on_mount(self) -> None:
        if not self._models:
            self._try_load_models()
        self._filtered = list(self._models)
        self._render_models()

    def _try_load_models(self) -> None:
        """Try to load available models from ollama."""
        try:
            import ollama

            response = ollama.list()
            self._models = [
                m.get("name", "") for m in response.get("models", []) if m.get("name")
            ]
        except Exception:
            logger.debug("Failed to load models from ollama", exc_info=True)

    def _render_models(self) -> None:
        try:
            container = self.query_one("#model-options", VerticalScroll)
            container.remove_children()
            if not self._filtered:
                container.mount(Static("[dim]No models found[/]"))
                return
            for i, name in enumerate(self._filtered):
                marker = " \u2713" if name == self._current else ""
                cls = "session-option" + (" -selected" if i == 0 else "")
                widget = Static(f"{name}{marker}", classes=cls, name=name)
                container.mount(widget)
            self._selected_index = 0
        except Exception:
            logger.debug("Failed to render models", exc_info=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        self._filtered = (
            [m for m in self._models if query in m.lower()]
            if query
            else list(self._models)
        )
        self._render_models()

    def key_up(self) -> None:
        self._move(-1)

    def key_down(self) -> None:
        self._move(1)

    def key_enter(self) -> None:
        if self._filtered and 0 <= self._selected_index < len(self._filtered):
            self.dismiss(self._filtered[self._selected_index])

    def _move(self, delta: int) -> None:
        options = list(self.query(".session-option"))
        if not options:
            return
        if 0 <= self._selected_index < len(options):
            options[self._selected_index].remove_class("-selected")
        self._selected_index = max(
            0, min(len(options) - 1, self._selected_index + delta)
        )
        options[self._selected_index].add_class("-selected")
        options[self._selected_index].scroll_visible()

    def on_static_click(self, event: Static.Click) -> None:
        widget = event.widget
        name = getattr(widget, "name", None)
        if name and name in self._filtered:
            self.dismiss(name)

    def action_cancel(self) -> None:
        self.dismiss("")
