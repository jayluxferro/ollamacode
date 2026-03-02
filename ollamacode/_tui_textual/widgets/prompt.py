from __future__ import annotations

import logging
from typing import Any

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

logger = logging.getLogger(__name__)


class PromptInput(TextArea):
    """Multi-line prompt input with submit on Enter, newline on Shift+Enter.

    Maintains an in-memory command history navigable with Up/Down when the
    cursor sits on the first or last line respectively.  The text the user
    was composing before entering history navigation is preserved and
    restored when they move past the newest history entry.
    """

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False),
        Binding("up", "history_prev", "Previous", show=False, priority=True),
        Binding("down", "history_next", "Next", show=False, priority=True),
    ]

    # -- Messages ----------------------------------------------------------------

    class Submitted(Message):
        """Posted when the user submits prompt text."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    # -- Lifecycle ---------------------------------------------------------------

    def __init__(
        self,
        placeholder: str = "How can I help you today?",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._placeholder = placeholder
        self._history: list[str] = []
        self._history_index: int = -1
        self._current_text: str = ""
        self._workspace_root: str = "."

    def on_mount(self) -> None:
        self.show_line_numbers = False

    # -- Rendering ---------------------------------------------------------------

    def render(self) -> object:
        """Show placeholder text when the input is empty."""
        if not self.text:
            from rich.text import Text

            return Text(self._placeholder, style="dim italic")
        return super().render()

    # -- Actions -----------------------------------------------------------------

    def action_submit(self) -> None:
        """Submit the current text and clear the input."""
        text = self.text.strip()
        if not text:
            return
        self._history.append(text)
        self._history_index = -1
        self.clear()
        self.post_message(self.Submitted(text))

    def action_history_prev(self) -> None:
        """Navigate to the previous history entry.

        Only activates when the cursor is on the first line so that normal
        cursor-up movement works in multi-line input.
        """
        if self.cursor_location[0] != 0:
            self.action_cursor_up()
            return
        if not self._history:
            return
        if self._history_index == -1:
            self._current_text = self.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        self.load_text(self._history[self._history_index])

    def action_history_next(self) -> None:
        """Navigate to the next history entry.

        Only activates when the cursor is on the last line so that normal
        cursor-down movement works in multi-line input.
        """
        if self.cursor_location[0] != self.document.line_count - 1:
            self.action_cursor_down()
            return
        if self._history_index == -1:
            return
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = -1
            self.load_text(self._current_text)
        else:
            self.load_text(self._history[self._history_index])

    # -- Properties --------------------------------------------------------------

    @property
    def workspace_root(self) -> str:
        """Root directory used for file-mention resolution."""
        return self._workspace_root

    @workspace_root.setter
    def workspace_root(self, value: str) -> None:
        self._workspace_root = value
