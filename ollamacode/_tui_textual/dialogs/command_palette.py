"""Command palette providers for Textual's built-in Ctrl+P palette."""

from __future__ import annotations

import logging
from functools import partial

from textual.command import Hit, Hits, Provider

logger = logging.getLogger(__name__)


class SessionCommands(Provider):
    """Command provider for session operations."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        # Static commands
        commands = [
            ("New Session", "Start a new chat session", "new_session"),
            ("List Sessions", "Browse past sessions", "list_sessions"),
            ("Clear Messages", "Clear current conversation", "clear_screen"),
        ]

        for title, help_text, action in commands:
            score = matcher.match(title)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(title),
                    partial(self._run_action, action),
                    help=help_text,
                )

        # Dynamic: search actual sessions
        if len(query) >= 2:
            try:
                from ollamacode.sessions import search_sessions

                sessions = search_sessions(query, limit=10)
                for s in sessions:
                    title = s.get("title", "Untitled")
                    sid = s.get("id", "")
                    display = f"Resume: {title}"
                    score = matcher.match(display)
                    if score > 0:
                        yield Hit(
                            score,
                            matcher.highlight(display),
                            partial(self._load_session, sid),
                            help=f"Session {sid[:8]}...",
                        )
            except Exception:
                pass

    def _run_action(self, action: str) -> None:
        self.app.run_action(action)

    def _load_session(self, session_id: str) -> None:
        from ..screens.session import SessionScreen

        app = self.app
        app.session_state.session_id = session_id
        app.switch_screen(SessionScreen(resume_session_id=session_id))


class ModelCommands(Provider):
    """Command provider for model switching."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        yield Hit(
            matcher.match("Change Model") or 0.1,
            matcher.highlight("Change Model"),
            self._show_model_picker,
            help="Switch the LLM model",
        )

        # Try to list models
        try:
            import ollama

            response = ollama.list()
            models = [
                m.get("name", "") for m in response.get("models", []) if m.get("name")
            ]
            for name in models:
                display = f"Model: {name}"
                score = matcher.match(display)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(display),
                        partial(self._set_model, name),
                        help=f"Switch to {name}",
                    )
        except Exception:
            pass

    def _show_model_picker(self) -> None:
        from ..dialogs.model_picker import ModelPickerDialog

        self.app.push_screen(ModelPickerDialog(current=self.app.model))

    def _set_model(self, name: str) -> None:
        self.app.model = name
        self.app.session_state.model = name
        self.app.notify(f"Model set to {name}")


class ThemeCommands(Provider):
    """Command provider for theme switching."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        from ..context.theme import list_themes

        for name in list_themes():
            display = f"Theme: {name}"
            score = matcher.match(display)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(display),
                    partial(self._set_theme, name),
                    help=f"Switch to {name} theme",
                )

    def _set_theme(self, name: str) -> None:
        from ..context.theme import generate_css, get_theme

        theme = get_theme(name)
        self.app.stylesheet.add_source(generate_css(theme), "theme")
        self.app.notify(f"Theme set to {name}")


class SlashCommands(Provider):
    """Command provider for slash commands."""

    COMMANDS = [
        ("/help", "Show available commands"),
        ("/model", "Change model"),
        ("/sessions", "List sessions"),
        ("/fix", "Run linter and send output"),
        ("/test", "Run tests and send output"),
        ("/plan", "Set a multi-step plan"),
        ("/continue", "Continue with current plan"),
        ("/auto", "Toggle autonomous mode"),
        ("/compact", "Toggle compact view"),
        ("/copy", "Copy last response to clipboard"),
        ("/summary", "Summarize conversation"),
        ("/quit", "Exit OllamaCode"),
    ]

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for cmd, desc in self.COMMANDS:
            score = matcher.match(cmd)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(cmd),
                    partial(self._run_slash, cmd),
                    help=desc,
                )

    def _run_slash(self, cmd: str) -> None:
        """Execute a slash command by posting it as a prompt."""
        from ..widgets.prompt import PromptInput

        self.app.post_message(PromptInput.Submitted(cmd))
