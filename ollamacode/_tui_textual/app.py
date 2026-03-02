"""OllamaCode TUI application -- Textual-based OpenCode-style interface."""

from __future__ import annotations

import logging
import os
from typing import Any

from textual.app import App
from textual.binding import Binding

from .context.state import AppState, SessionState
from .context.theme import generate_css, get_theme

logger = logging.getLogger(__name__)


class OllamaCodeApp(App):
    """Main OllamaCode TUI application."""

    CSS_PATH = "styles.tcss"

    TITLE = "OllamaCode"

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New Session", show=True),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=False),
        Binding("ctrl+backslash", "toggle_sidebar", "Sidebar", show=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    # Application state
    app_state: AppState
    session_state: SessionState

    def __init__(
        self,
        *,
        session: Any = None,
        model: str = "",
        system_extra: str = "",
        workspace_root: str | None = None,
        provider: Any = None,
        provider_name: str = "ollama",
        theme_name: str = "opencode",
        session_id: str | None = None,
        session_title: str | None = None,
        session_history: list[dict[str, Any]] | None = None,
        confirm_tool_calls: bool = False,
        autonomous_mode: bool = False,
        max_tool_rounds: int = 20,
        max_messages: int = 0,
        max_tool_result_chars: int = 0,
        linter_command: str | None = None,
        test_command: str | None = None,
        docs_command: str | None = None,
        profile_command: str | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.mcp_session = session
        self.model = model
        self.system_extra = system_extra
        self.provider = provider
        self.provider_name = provider_name
        self.confirm_tool_calls = confirm_tool_calls
        self.autonomous_mode = autonomous_mode
        self.max_tool_rounds = max_tool_rounds
        self.max_messages = max_messages
        self.max_tool_result_chars = max_tool_result_chars
        self.linter_command = linter_command
        self.test_command = test_command
        self.docs_command = docs_command
        self.profile_command = profile_command

        self.app_state = AppState(
            workspace_root=workspace_root or os.getcwd(),
        )
        self.session_state = SessionState(
            session_id=session_id or "",
            title=session_title or "",
            model=model,
            provider_name=provider_name,
            autonomous=autonomous_mode,
        )
        self.session_history: list[dict[str, Any]] = session_history or []
        self._theme_name = theme_name

    def on_mount(self) -> None:
        """Set up theme and push initial screen."""
        # Apply theme CSS
        theme = get_theme(self._theme_name)
        self.stylesheet.add_source(generate_css(theme), "theme")

        # Push home or session screen
        from .screens.home import HomeScreen
        from .screens.session import SessionScreen

        if self.session_state.session_id:
            self.push_screen(SessionScreen())
        else:
            self.push_screen(HomeScreen())

    def action_new_session(self) -> None:
        """Start a new session."""
        from .screens.home import HomeScreen

        self.session_state = SessionState(
            model=self.model,
            provider_name=self.provider_name,
        )
        self.session_history = []
        self.switch_screen(HomeScreen())

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        try:
            from .widgets.sidebar import Sidebar

            sidebar = self.query_one(Sidebar)
            sidebar.toggle_class("-hidden")
        except Exception:
            logger.debug("Sidebar not found or toggle failed", exc_info=True)

    def action_cancel_generation(self) -> None:
        """Cancel any running generation."""
        self.session_state.is_busy = False
        self.session_state.is_streaming = False
        # Workers will check is_cancelled

    def action_clear_screen(self) -> None:
        """Clear the current screen."""
        from .screens.session import SessionScreen

        screen = self.screen
        if isinstance(screen, SessionScreen):
            screen.clear_messages()
