"""Message list and individual message widgets for the chat view."""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual.reactive import reactive

logger = logging.getLogger(__name__)


class MessageList(VerticalScroll):
    """Scrollable container for chat messages."""

    def scroll_to_latest(self) -> None:
        """Scroll to the bottom to show latest message."""
        self.scroll_end(animate=False)


class UserMessage(Widget):
    """A user message with left border accent."""

    def __init__(self, text: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(self._text, classes="user-message-text", markup=False)


class AssistantMessage(Widget):
    """An assistant message that supports streaming markdown and tool calls."""

    is_streaming = reactive(False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._markdown_widget: Markdown | None = None
        self._accumulated_text: str = ""

    def compose(self) -> ComposeResult:
        self._markdown_widget = Markdown("", classes="assistant-markdown")
        yield self._markdown_widget

    async def append_text(self, chunk: str) -> None:
        """Append streaming text chunk to the markdown widget."""
        self._accumulated_text += chunk
        if self._markdown_widget is not None:
            try:
                await self._markdown_widget.update(self._accumulated_text)
            except Exception:
                logger.debug("Failed to update markdown", exc_info=True)

    async def add_tool_call(self, widget: Widget) -> None:
        """Mount a tool call widget after the markdown content."""
        await self.mount(widget)

    def get_text(self) -> str:
        """Return the accumulated text content."""
        return self._accumulated_text

    async def finalize(self) -> None:
        """Called when streaming is complete."""
        self.is_streaming = False
        if self._markdown_widget is not None and self._accumulated_text:
            try:
                await self._markdown_widget.update(self._accumulated_text)
            except Exception:
                logger.debug("Failed to finalize markdown", exc_info=True)
