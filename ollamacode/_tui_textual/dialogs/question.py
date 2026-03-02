"""Question dialog — agent asks the user a question."""

from __future__ import annotations


from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class QuestionDialog(ModalScreen[str]):
    """Modal for answering an agent's question."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._question = question
        self._options = options or []

    def compose(self) -> ComposeResult:
        with Vertical(id="question-container"):
            yield Static("[bold]Agent Question[/]")
            yield Static(self._question, id="question-text")
            if self._options:
                with Horizontal(id="question-options"):
                    for i, opt in enumerate(self._options):
                        yield Button(
                            opt,
                            id=f"qopt-{i}",
                            variant="primary" if i == 0 else "default",
                        )
            yield Static("[dim]Or type a custom answer:[/]")
            yield Input(placeholder="Your answer...", id="question-input")
            with Horizontal(id="question-buttons"):
                yield Button("Submit", variant="success", id="question-submit")
                yield Button("Skip", variant="default", id="question-skip")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("qopt-"):
            try:
                idx = int(bid.removeprefix("qopt-"))
                if 0 <= idx < len(self._options):
                    self.dismiss(self._options[idx])
            except ValueError:
                pass
        elif bid == "question-submit":
            text = self.query_one("#question-input", Input).value.strip()
            self.dismiss(text if text else "(no answer)")
        elif bid == "question-skip":
            self.dismiss("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.dismiss(text if text else "(no answer)")

    def action_cancel(self) -> None:
        self.dismiss("")
