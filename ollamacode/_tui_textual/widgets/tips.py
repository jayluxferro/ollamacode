from __future__ import annotations

import random

from textual.widgets import Static

TIPS: list[str] = [
    "Use **@** to mention files in your prompt",
    "Press **Ctrl+P** to open the command palette",
    "Use **/help** for a list of slash commands",
    "Press **Ctrl+N** to start a new session",
    "Press **Ctrl+\\\\** to toggle the sidebar",
    "Use **/model** to switch between models",
    "Press **Tab** to switch between Build and Plan mode",
    "Use **/sessions** to browse past conversations",
    "Press **Escape** to cancel a running generation",
    "Use **!command** to run shell commands directly",
    "Use **/fix** to run your linter and send output to the model",
    "Use **/test** to run tests and send output to the model",
    "Press **Ctrl+L** to clear the screen",
    "Use **/compact** to toggle compact view",
    "Use **/auto** to enable autonomous mode",
]


class Tips(Static):
    """Display a random tip to the user."""

    def on_mount(self) -> None:
        self.refresh_tip()

    def refresh_tip(self) -> None:
        """Pick a new random tip and update the display."""
        tip = random.choice(TIPS)
        self.update(f"Tip: {tip}")
