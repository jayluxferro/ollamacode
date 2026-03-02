"""Session header bar — title, tokens, cost, model."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


class SessionHeader(Widget):
    """Top bar showing session info."""

    title = reactive("New Session")
    token_count = reactive(0)
    cost = reactive(0.0)
    model_name = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(self.title, id="header-title")
        yield Static("", id="header-tokens")
        yield Static("", id="header-cost")
        yield Static("", id="header-model")

    def watch_title(self, value: str) -> None:
        try:
            self.query_one("#header-title", Static).update(value)
        except Exception:
            pass

    def watch_token_count(self, value: int) -> None:
        try:
            if value > 0:
                if value >= 1000:
                    display = f"{value / 1000:.1f}K tokens"
                else:
                    display = f"{value} tokens"
                self.query_one("#header-tokens", Static).update(display)
        except Exception:
            pass

    def watch_cost(self, value: float) -> None:
        try:
            if value > 0:
                self.query_one("#header-cost", Static).update(f"${value:.4f}")
        except Exception:
            pass

    def watch_model_name(self, value: str) -> None:
        try:
            self.query_one("#header-model", Static).update(value)
        except Exception:
            pass
