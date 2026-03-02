"""Session list dialog — browse and select sessions."""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

logger = logging.getLogger(__name__)


class SessionOption(Static):
    """A single session entry in the list."""

    def __init__(self, session_id: str, title: str, info: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        self._title = title
        self._info = info

    def render(self) -> str:
        title = self._title or "(untitled)"
        return f"[bold]{title[:50]}[/]\n[dim]{self._info}[/]"


class SessionListDialog(ModalScreen[str]):
    """Modal for browsing and selecting sessions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sessions: list[dict[str, Any]] = []
        self._selected_index: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="session-list-container"):
            yield Static("[bold]Sessions[/]", id="session-list-title")
            yield Input(placeholder="Search sessions...", id="session-search")
            yield VerticalScroll(id="session-options")

    def on_mount(self) -> None:
        self._load_sessions()

    def _load_sessions(self, query: str = "") -> None:
        """Load sessions from database."""
        try:
            from ollamacode.sessions import list_sessions, search_sessions

            if query:
                self._sessions = search_sessions(query, limit=30)
            else:
                self._sessions = list_sessions(limit=30)
        except Exception:
            logger.debug("Failed to load sessions", exc_info=True)
            self._sessions = []
        self._render_sessions()

    def _render_sessions(self) -> None:
        """Render session list."""
        try:
            container = self.query_one("#session-options", VerticalScroll)
            container.remove_children()
            if not self._sessions:
                container.mount(Static("[dim]No sessions found[/]"))
                return
            for i, s in enumerate(self._sessions):
                sid = s.get("id", "")
                title = s.get("title", "")
                updated = str(s.get("updated_at", ""))[:19]
                count = s.get("message_count", 0)
                info = f"{sid[:8]}... | {count} msgs | {updated}"
                option = SessionOption(
                    sid,
                    title,
                    info,
                    classes="session-option" + (" -selected" if i == 0 else ""),
                )
                container.mount(option)
            self._selected_index = 0
        except Exception:
            logger.debug("Failed to render sessions", exc_info=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._load_sessions(event.value)

    def on_static_click(self, event: Static.Click) -> None:
        """Handle click on a session option."""
        widget = event.widget
        if isinstance(widget, SessionOption):
            self.dismiss(widget.session_id)

    def key_up(self) -> None:
        self._move_selection(-1)

    def key_down(self) -> None:
        self._move_selection(1)

    def key_enter(self) -> None:
        options = self.query(".session-option")
        if options and 0 <= self._selected_index < len(options):
            option = options[self._selected_index]
            if isinstance(option, SessionOption):
                self.dismiss(option.session_id)

    def _move_selection(self, delta: int) -> None:
        options = list(self.query(".session-option"))
        if not options:
            return
        # Remove old selection
        if 0 <= self._selected_index < len(options):
            options[self._selected_index].remove_class("-selected")
        self._selected_index = max(
            0, min(len(options) - 1, self._selected_index + delta)
        )
        options[self._selected_index].add_class("-selected")
        options[self._selected_index].scroll_visible()

    def action_cancel(self) -> None:
        self.dismiss("")
