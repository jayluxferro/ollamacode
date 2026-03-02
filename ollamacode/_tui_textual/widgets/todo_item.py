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

    def __init__(self, text: str, done: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.todo_text = text
        self.done = done

    def render(self) -> str:
        check = "\u2611" if self.done else "\u2610"
        style = "[dim strike]" if self.done else ""
        end = "[/]" if self.done else ""
        return f"{check} {style}{self.todo_text}{end}"
