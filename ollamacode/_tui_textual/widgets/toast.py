"""Toast notification system for the OllamaCode Textual TUI.

Provides auto-dismissing notifications that stack in the top-right corner,
inspired by OpenCode's toast system.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

logger = logging.getLogger(__name__)

# Variant → border colour
VARIANT_COLORS: dict[str, str] = {
    "info": "#89b4fa",
    "success": "#a6e3a1",
    "warning": "#fab387",
    "error": "#f38ba8",
}

VARIANT_ICONS: dict[str, str] = {
    "info": "\u2139",  # info symbol
    "success": "\u2713",  # check mark
    "warning": "\u26a0",  # warning triangle
    "error": "\u2717",  # cross mark
}


class ToastItem(Widget):
    """A single toast notification."""

    DEFAULT_CSS = """
    ToastItem {
        width: 50;
        max-width: 60;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        layer: toast;
    }
    """

    def __init__(
        self,
        message: str,
        *,
        title: str = "",
        variant: str = "info",
        duration: float = 5.0,
        id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id=id, **kwargs)
        self._message = message
        self._title = title
        self._variant = variant
        self._duration = duration
        self._color = VARIANT_COLORS.get(variant, VARIANT_COLORS["info"])
        self._icon = VARIANT_ICONS.get(variant, VARIANT_ICONS["info"])

    def compose(self) -> ComposeResult:
        header = f"{self._icon} {self._title}" if self._title else self._icon
        yield Static(header, classes="toast-header")
        yield Static(self._message, classes="toast-body")

    def on_mount(self) -> None:
        """Start auto-dismiss timer."""
        if self._duration > 0:
            self.set_timer(self._duration, self._dismiss)

    def _dismiss(self) -> None:
        """Remove this toast."""
        try:
            self.remove()
        except Exception:
            pass


class ToastContainer(Widget):
    """Container that holds toast notifications, docked to top-right."""

    DEFAULT_CSS = """
    ToastContainer {
        dock: right;
        width: 52;
        height: auto;
        max-height: 50%;
        layer: toast;
        padding: 1;
        overflow-y: hidden;
    }
    """

    def show(
        self,
        message: str,
        *,
        title: str = "",
        variant: str = "info",
        duration: float = 5.0,
    ) -> None:
        """Show a new toast notification."""
        toast = ToastItem(
            message, title=title, variant=variant, duration=duration
        )
        self.mount(toast)

    def error(self, message: str, *, title: str = "Error") -> None:
        """Show an error toast."""
        self.show(message, title=title, variant="error", duration=8.0)

    def success(self, message: str, *, title: str = "") -> None:
        """Show a success toast."""
        self.show(message, title=title, variant="success", duration=4.0)

    def warning(self, message: str, *, title: str = "") -> None:
        """Show a warning toast."""
        self.show(message, title=title, variant="warning", duration=6.0)

    def info(self, message: str, *, title: str = "") -> None:
        """Show an info toast."""
        self.show(message, title=title, variant="info", duration=5.0)
