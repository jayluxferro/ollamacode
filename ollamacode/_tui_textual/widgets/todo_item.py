"""Todo item widget for sidebar."""

from __future__ import annotations

from typing import Any

from textual.widget import Widget


class TodoItem(Widget):
    """Single todo item display."""

    DEFAULT_CSS = """
    TodoItem {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        text: str,
        done: bool = False,
        status: str = "pending",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.todo_text = text
        self.done = done
        self.status = "completed" if done else status

    def render(self) -> str:
        if self.status == "completed":
            check = "\u2611"
            style = "[dim strike]"
            end = "[/]"
        elif self.status == "in_progress":
            check = "\u25d0"
            style = ""
            end = ""
        elif self.status == "cancelled":
            check = "\u2298"
            style = "[dim]"
            end = "[/]"
        else:
            check = "\u2610"
            style = ""
            end = ""
        return f"{check} {style}{self.todo_text}{end}"
