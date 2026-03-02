"""Home screen -- centered logo, prompt, and tips."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Static

from ..widgets.logo import Logo
from ..widgets.prompt import PromptInput
from ..widgets.tips import Tips


class HomeScreen(Screen):
    """Landing screen with centered logo and prompt input."""

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="home-container"):
            yield Static("", id="home-spacer-top")
            yield Logo(id="home-logo")
            with Center():
                with Vertical(id="home-prompt-container"):
                    yield PromptInput(id="home-prompt")
                    yield Tips(id="home-tips")
            yield Static("", id="home-spacer-bottom")
        # Footer
        app = self.app
        ws = getattr(app, "app_state", None)
        workspace = ws.workspace_root if ws else "."
        model = getattr(app, "model", "")
        provider = getattr(app, "provider_name", "ollama")
        footer_text = f"  {workspace}  |  {provider}/{model}  |  v1.0.0"
        yield Static(footer_text, id="home-footer")

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle prompt submission -- create session and switch to session screen."""
        from ..screens.session import SessionScreen

        app = self.app
        # Create session
        try:
            from ollamacode.sessions import create_session

            session_id = create_session(
                title=event.text[:60],
                workspace_root=app.app_state.workspace_root,
            )
            app.session_state.session_id = session_id
            app.session_state.title = event.text[:60]
        except Exception:
            import uuid

            app.session_state.session_id = str(uuid.uuid4())
            app.session_state.title = event.text[:60]

        # Switch to session screen with the initial prompt
        session_screen = SessionScreen(initial_prompt=event.text)
        app.switch_screen(session_screen)
